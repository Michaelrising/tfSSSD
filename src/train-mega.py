import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from utils.util import  get_mask_holiday
from imputers.MegaModel import MegaImputer


def train(output_directory,
          log_directory,
          learning_rate,
          batch_size,
          epochs,
          alg=None,
          stock=None,
          seq_len=200,
          ):
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    print('=' * 50)
    print("=" * 23 + 'Mega' + "=" * 22)
    print('=' * 50)

    # generate experiment (local) path
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    local_path = '/' + current_time + '_seq_{}_Layers_{}'.format(seq_len, model_config['depth'])

    # Get shared output_directory ready
    output_directory = output_directory + local_path
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    train_log_dir = log_directory + local_path
    if not os.path.isdir(train_log_dir):
        os.makedirs(train_log_dir)
        os.chmod(output_directory, 0o775)
    print("log directory", train_log_dir, flush=True)
    ### Custom data loading and reshaping ###

    # prepare X and Y for model.fit()

    training_data = []
    training_mask =[]
    for ticker in ['DJ', 'ES', 'SE']:
        data_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
        ticker_data = np.load('../datasets/Stocks/' + data_name, allow_pickle=True).astype(np.float32) # L N C
        scalar0 = MinMaxScaler()
        ticker_data = np.array([scalar0.fit_transform(tk) for tk in ticker_data.transpose([1, 0, 2])]).transpose([1, 0, 2])  # N L C -> L N C
        # generate masks: observed_masks + man_made mask
        ticker_mask = get_mask_holiday(ticker_data, ratio=2)  # N L C
        ticker_mask = ticker_mask.numpy().transpose([1, 0, 2]) # L N C
        # ticker_mask = (~np.isnan(ticker_data)).astype(np.float32)
        # for i in range(0, ticker_data.shape[0] - args.seq_len, args.mw):
        #     ticker_chunk = ticker_data[i:i + args.seq_len]  # L N C
        #     training_data.append(ticker_chunk)
        #     training_mask.append(ticker_mask[i:i + args.seq_len])
        for i in range(ticker_data.shape[0] // args.seq_len):
            ticker_chunk = ticker_data[args.seq_len * i:args.seq_len * (i + 1)] # L N C
            training_data.append(ticker_chunk)
            training_mask.append(ticker_mask[args.seq_len * i:args.seq_len * (i + 1)])
        np.save(train_log_dir +'/' + ticker + '_all_stocks_2013-01-02_to_2023-01-01_gt_masks.npy', ticker_mask)
    training_data = np.concatenate(training_data, axis=1).astype(float)  # L B K
    training_mask = np.concatenate(training_mask, axis=1).astype(np.float32) # L B K
    training_all = np.concatenate((training_data.transpose([1,0,2]), training_mask.transpose([1,0,2])), axis=-1) # B L 2*K
    # shuffle data
    np.random.shuffle(training_all)
    training_data = training_all[..., :5].transpose([1, 0, 2]) # L B K
    training_mask = training_all[..., 5:].transpose([1, 0, 2]) # L B K
    print(training_data.shape)

    print('Data loaded')

    L, N, K = training_data.shape  # C is the dimension of each audio, L is audio length, N is the audio batch

    # generate masks: observed_masks + man_made mask
    observed_mask = (~np.isnan(training_data)).astype(np.float32)
    observed_mask = tf.transpose(tf.convert_to_tensor(observed_mask, tf.float32),  perm=[1, 2, 0])
    training_data = tf.transpose(tf.convert_to_tensor(np.nan_to_num(training_data), dtype=tf.float32), perm=[1, 2, 0])  # batch dim # L N C -> [N C L]
    print("missing rate:" + str(1-np.sum(training_mask)/(N*K*L)))
    training_mask = tf.transpose(tf.convert_to_tensor(training_mask, dtype=tf.float32), perm=[1, 2, 0]) # batch dim # L N C -> [N C L]
    loss_mask = tf.cast(observed_mask - training_mask, tf.bool) #tf.cast(observed_mask - training_mask, tf.bool)

    assert training_data.shape == training_mask.shape == loss_mask.shape

    model = MegaImputer(**model_config)

    ############ Save config file #######


    # prepare X & Y
    training_data = tf.transpose(training_data, perm=[0, 2, 1])
    training_mask = tf.transpose(training_mask, perm=[0, 2, 1])
    loss_mask =  tf.transpose(loss_mask, perm=[0, 2, 1])

    X = [training_data,training_mask ,loss_mask]
    Y = training_data

    # define optimizer
    p1 = int(0.5 * epochs * N//batch_size)
    p2 = int(0.75 * epochs * N//batch_size)
    # p3 = int(0.8 * self.epochs * series.shape[0] / self.batch_size)
    boundaries = [p1]
    values = [learning_rate, learning_rate * 0.1]

    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-6)

    # define loss

    loss = keras.losses.MeanSquaredError()
    # define callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)
    earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=output_directory,
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True,
                                          )

    # training
    model.compile(optimizer=optimizer, loss=loss)

    # model.train_step(([training_data[:16], training_mask[:16], loss_mask[:16]], training_data[:16]))
    history = model.fit(
        x=X,
        y=Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[tensorboard_callback,
                   earlyStop_loss_callback,
                   best_checkpoint_callback],
    )
    plt.plot(history.history["loss"], c='blue', label='Loss')
    plt.plot(history.history["val_loss"], c='orange', label='Val_loss')
    plt.grid()
    plt.legend()
    plt.title("Training Loss")
    plt.savefig(train_log_dir + '/training.png')
    plt.show()
    model.summary()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='/config/config_SSSD_stocks.json',
    #                     help='JSON file for configuration')
    parser.add_argument('-ignore_warning', type=str, default=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--stock', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if args.ignore_warning:
        # Disable absl INFO and WARNING log messages
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    global model_config
    model_config = {}
    model_config['in_feature'] = 5
    model_config['mid_features'] = 128  # original name is dim
    model_config['depth'] = 8
    model_config['out_features'] = 5
    model_config['chunk_size'] = -1
    model_config['ff_mult'] = 2
    model_config['pre_norm'] = True

    output_directory = '../results/stocks/' + args.stock + "/Mega/"
    log_directory = '../log/stocks/' + args.stock + "/Mega/"
    learning_rate = 1e-3
    train(output_directory,
          log_directory,
          learning_rate,
          args.batch_size,
          args.epochs,
          args.seq_len,
          )
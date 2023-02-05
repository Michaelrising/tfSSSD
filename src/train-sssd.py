import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.optimizers.legacy import Adam
import sys
from datetime import datetime
from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, std_normal, get_mask_holiday

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDImputer import SSSDImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from einops import rearrange
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def simple_imputer(x):
    masks = np.isnan(x[:, 0])
    index = np.where(masks)[0]
    imputation = []
    for d in index:
        choice = np.array([d - 4, d - 3, d - 2, d - 1, d + 1, d + 2, d + 3, d + 4, d + 5])
        choice = choice[(choice > 0) * (choice < x.shape[0] - 1)]
        m = ~np.isin(choice, index)
        choice = choice[m]
        dd = x[choice].reshape(-1, x.shape[1])
        imputation.append(np.mean(dd, axis=0))
    x[masks] = np.array(imputation)
    return x


# @tf.function
def train(output_directory,
          log_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k,
          missing_rate,
          batch_size,
          epochs,
          alg=None,
          stock=None,
          seq_len=1000,
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
    if alg == 'S4':
        print('=' * 50)
        print("=" * 22 + 'SSSD-S4' + "=" * 21)
        print('=' * 50)
    elif alg == 'S5':
        print('=' * 50)
        print("=" * 22 + 'SSSD-S5' + "=" * 21)
        print('=' * 50)
    elif alg == 'Mega':
        print('=' * 50)
        print("=" * 21 + 'SSSD-Mega' + "=" * 20)
        print('=' * 50)

    # generate experiment (local) path
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    local_path = stock + '/SSSD-' + alg + '/' + current_time + '_seq_{}_T_{}_Layers_{}'.format(seq_len,diffusion_config['T'], model_config['num_res_layers'])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    train_log_dir = log_directory + stock + '/SSSD-' + alg + '/' + current_time + '_seq_{}_T_{}_Layers_{}'.format(
        seq_len, diffusion_config['T'], model_config['num_res_layers'])
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
        ticker_mask = get_mask_holiday(ticker_data)  # N L C
        ticker_mask = ticker_mask.numpy().transpose([1, 0, 2]) # L N C
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
    print('Loading stocks data: ' + stock)
    print(training_data.shape)

    print('Data loaded')

    L, N, K = training_data.shape  # C is the dimension of each audio, L is audio length, N is the audio batch

    # generate masks: observed_masks + man_made mask
    # observed_mask = (~np.isnan(training_data)).astype(np.float32)
    training_data = tf.transpose(tf.convert_to_tensor(np.nan_to_num(training_data), dtype=tf.float32), perm=[1, 2, 0])  # batch dim # L N C -> [N C L]
    print("missing rate:" + str(1-np.sum(training_mask)/(N*K*L)))
    training_mask = tf.transpose(tf.convert_to_tensor(training_mask, dtype=tf.float32), perm=[1, 2, 0]) # batch dim # L N C -> [N C L]
    if only_generate_missing:
        loss_mask = tf.cast(1.-training_mask, tf.bool)
    else:
        loss_mask = tf.cast(tf.ones_like(training_mask), tf.bool)

    assert training_data.shape == training_mask.shape == loss_mask.shape

    model_config['s4_lmax'] = L

    if use_model == 0:
        net = DiffWaveImputer(**model_config)
    elif use_model == 1:
        net = SSSDSAImputer(**model_config)
    elif use_model == 2:
        net = SSSDImputer(**model_config, alg=alg)
    else:
        print('Model chosen not available.')
        net = None

    ############ Save config file #######
    config[ "diffusion_config"] = diffusion_config
    config['wavenet_config'] = model_config
    config['train_config'] = train_config

    config_filename = train_log_dir + '/config_SSSD_stocks_seq_{}_T_{}_Layers_{}'.format(seq_len,diffusion_config['T'], model_config['num_res_layers'])
    print('configuration file name:', config_filename)
    with open(config_filename + ".json", "w") as f:
        json.dump(config, f, indent=4)


    # prepare X
    noise = tf.random.normal(shape=tf.shape(training_data), dtype=training_data.dtype)
    diffusion_steps = tf.random.uniform(shape=(tf.shape(training_data)[0], 1, 1), maxval=model_config['T'], dtype=tf.int32)  # randomly sample diffusion steps from 1~T
    X = [noise, training_data, training_mask, loss_mask, diffusion_steps]
    # define optimizer
    p1 = int(0.5 * epochs * N//batch_size)
    p2 = int(0.75 * epochs * N//batch_size)
    # p3 = int(0.8 * self.epochs * series.shape[0] / self.batch_size)
    boundaries = [p1]
    values = [learning_rate, learning_rate * 0.1]

    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-6, amsgrad=True)

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
    net.compile(optimizer=optimizer, loss=loss)
    history = net.fit(
        x=X,
        y=None,
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
    net.summary()

    # np.save(log_directory + 'SSSD-' + alg + '/' + current_time + '/observed_data.npy', training_data.numpy())

    # np.save(train_log_dir + '/stock_data_seq_len_'+str(seq_len) +'.npy', training_data.numpy())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/config/config_SSSD_stocks.json',
                        help='JSON file for configuration')
    parser.add_argument('-ignore_warning', type=str, default=True)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--alg', type=str, default='S4')
    parser.add_argument('--stock', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=400)
    parser.add_argument('--num_layers', type=int, default=18)
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

    sys.path.append(os.getcwd() + args.config)
    with open(os.getcwd() + args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)

    config['train_config']["alg"] = args.alg
    config['train_config']["stock"] = args.stock
    config['train_config']["batch_size"] = args.batch_size
    config['train_config']["seq_len"] = args.seq_len


    print(config)

    train_config = config["train_config"]  # training parameters


    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']
    model_config['num_res_layers'] = args.num_layers

    tf.debugging.set_log_device_placement(True)
    train(**train_config)

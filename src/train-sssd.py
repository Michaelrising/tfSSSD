import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
from datetime import datetime
from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, std_normal, get_mask_holiday

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDImputer import SSSDImputer
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
    elif alg == 'transformer':
        print('=' * 50)
        print("=" * 17 + 'SSSD-TransFormer' + "=" * 17)
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
    local_path = stock + '/SSSD-' + alg + '/' + current_time + '/sssd_model'

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    ### Custom data loading and reshaping ###

    # prepare X and Y for model.fit()

    data_name = 'scaled_' + stock + '_all_stocks_2013-01-02_to_2023-01-01.npy'
    training_data = np.load(trainset_config['train_data_path'] + data_name)
    print('Loading stocks data: ' + stock )
    print(training_data.shape)

    print('Data loaded')

    B, L, C = training_data.shape  # B is batchsize, C is the dimension of each audio, L is audio length
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
    # print_size(net.summary())


    # set up log writer

    train_log_dir = log_directory + stock +'/SSSD-' + alg + '/' + current_time + '/sssd_log'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = -1  # find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}'.format(ckpt_iter))
            net = keras.models.load_model(model_path)

            # feed model dict and optimizer state
            # net.load_state_dict(checkpoint['model_state_dict'])
            # if 'optimizer_state_dict' in checkpoint:
            #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')




    if masking == 'rm':
        mask = np.ones((C, L))
        # mask = tf.Variable(mask_array, trainable=False)
        length_index = np.arange(mask.shape[0])  # lenght of series indexes
        for channel in range(mask.shape[1]):
            # perm = torch.randperm(len(length_index))
            perm = np.random.permutation(len(length_index))

            sample_num = int(mask.shape[0] * missing_rate)
            idx = perm[0:sample_num]
            mask[:, channel][idx] = 0
        mask = tf.transpose(tf.convert_to_tensor(mask, dtype=tf.float32), perm=[1, 0])
        mask = tf.tile(tf.expand_dims(mask, 0), [B, 1, 1])
    # elif masking == 'mnr':
    #     mask = get_mask_mnr(training_data, missing_k)
    # elif masking == 'bm':
    #     mask = get_mask_bm(training_data, missing_k)
    elif masking == 'holiday':
        mask = get_mask_holiday(training_data, batch_size)
    training_data = tf.convert_to_tensor(np.nan_to_num(training_data), dtype=tf.float32)
    # loss_mask = tf.logical_not(mask)  # .bool()
    training_data = tf.transpose(training_data, perm=[0, 2, 1])  # batch dim = [B, C, L]
    mask = tf.transpose(mask, perm=[0, 2, 1])
    # loss_mask = tf.transpose(loss_mask, perm=[0, 2, 1])

    assert training_data.shape == mask.shape # == loss_mask.shape

    # prepare Y
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], tf.cast(_dh["Alpha_bar"], tf.float32)

    diffusion_steps = tf.random.uniform(shape=(B,), maxval=T, dtype=tf.int32)  # randomly sample diffusion steps from 1~T

    z = std_normal(training_data.shape)
    if only_generate_missing == 1:
        z = training_data * mask + z * (1. - mask)
    transformed_X = tf.cast(tf.math.sqrt(tf.reshape(tf.gather(Alpha_bar, diffusion_steps), shape=[B, 1, 1])),
                            dtype=training_data.dtype) * training_data + tf.cast(tf.math.sqrt(
        1 - tf.reshape(tf.gather(Alpha_bar, diffusion_steps), shape=[B, 1, 1])),
        dtype=z.dtype) * z  # compute x_t from q(x_t|x_0)

    X = [tf.cast(transformed_X, dtype=tf.float32), tf.cast(training_data, dtype=tf.float32), mask, tf.reshape(diffusion_steps, shape=(B, 1))]

    # define optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-6, amsgrad=True, clipnorm=0.5)
    # define loss
    loss = keras.losses.MeanSquaredError()
    # define callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)
    earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=5)
    earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=5)
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=output_directory,
                                            save_weights_only=False,
                                            monitor='accuracy',
                                            mode='max',
                                            save_best_only=True,
                                            save_format='tf',
                                          )

    # training
    net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = net.fit(
        x=X,
        y=z,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[tensorboard_callback,
                   earlyStop_loss_callback,
                   earlyStop_accu_call_back,
                   best_checkpoint_callback],
    )
    plt.plot(history.history["loss"], c='blue', lable='Loss')
    plt.plot(history.history["val_loss"], c='orange', lable='Val_loss')
    plt.plot(history.history["accuracy"], c='red', lable='Accuracy')
    plt.grid()
    plt.legend()
    plt.title("Training Loss and Accuracy")
    plt.savefig(train_log_dir + '/training.png')
    plt.show()
    net.summary()

    # np.save(log_directory + 'SSSD-' + alg + '/' + current_time + '/observed_data.npy', training_data.numpy())
    np.save(train_log_dir + '/gt_mask.npy', mask.numpy())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/config/config_SSSD_stocks.json',
                        help='JSON file for configuration')
    parser.add_argument('-ignore_warning', type=str, default=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--alg', type=str, default='S4')
    parser.add_argument('--stock', type=str, default='DJ')
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

    config = json.loads(data)
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

    tf.debugging.set_log_device_placement(True)
    train(**train_config,
          alg=args.alg,
          stock=args.stock)

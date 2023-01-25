import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import datetime
from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, std_normal

# from imputers.DiffWaveImputer import DiffWaveImputer
# from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDImputer import SSSDS4Imputer

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
          device):
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

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    # for key in diffusion_hyperparams:
    #     if key != "T":
    #         diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)
    # predefine model
    with tf.device(device):
        if use_model == 0:
            net = DiffWaveImputer(**model_config)  # .to(device)
        elif use_model == 1:
            net = SSSDSAImputer(**model_config)  # .to(device)
        elif use_model == 2:
            net = SSSDS4Imputer(**model_config)  # .to(device)
        else:
            print('Model chosen not available.')
            net = None
    # print_size(net.summary())


    # set up log writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = log_directory + "/" + local_path + '/' + current_time
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

    ### Custom data loading and reshaping ###

    # prepare X and Y for model.fit()
    training_data = np.load(trainset_config['train_data_path'])
    with tf.device(device):
        training_data = tf.convert_to_tensor(training_data, dtype=tf.float32)
    print('Data loaded')
    B, C, L = training_data.shape  # B is batchsize, C is the dimension of each audio, L is audio length

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
    elif masking == 'mnr':
        transposed_mask = get_mask_mnr(training_data, missing_k)
    elif masking == 'bm':
        transposed_mask = get_mask_bm(training_data, missing_k)

    mask = tf.transpose(tf.convert_to_tensor(mask, dtype=tf.float32), perm=[1, 0])
    mask = tf.tile(tf.expand_dims(mask, 0), [B, 1, 1])  # .float().to(device)
    loss_mask = tf.logical_not(tf.cast(mask, dtype=tf.bool))  # .bool()
    training_data = tf.transpose(training_data, perm=[0, 2, 1])  # batch dim = [B, C, L]

    assert training_data.shape == mask.shape == loss_mask.shape

    # prepare Y
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], tf.cast(_dh["Alpha_bar"], tf.float32)

    with tf.device(device):
        diffusion_steps = tf.random.uniform(shape=(B,), maxval=T,
                                            dtype=tf.int32)  # randomly sample diffusion steps from 1~T

    z = std_normal(training_data.shape, device)
    if only_generate_missing == 1:
        z = training_data * mask + z * (1. - mask)
    transformed_X = tf.cast(tf.math.sqrt(tf.reshape(tf.gather(Alpha_bar, diffusion_steps), shape=[B, 1, 1])),
                            dtype=training_data.dtype) * training_data + tf.cast(tf.math.sqrt(
        1 - tf.reshape(tf.gather(Alpha_bar, diffusion_steps), shape=[B, 1, 1])),
        dtype=z.dtype) * z  # compute x_t from q(x_t|x_0)

    X = [tf.cast(transformed_X, dtype=tf.float32), tf.cast(training_data, dtype=tf.float32), mask, tf.reshape(diffusion_steps, shape=(B, 1))]

    # define optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # define loss
    loss = keras.losses.MeanSquaredError()
    # define metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)
    earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)
    earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=3)
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=output_directory + "/" + local_path + '/' + current_time,
                                            save_weights_only=False,
                                            monitor='accuracy',
                                            mode='max',
                                            save_best_only=True,
                                          )

    # training
    # net.build(input_shape=((model_config["in_channels"], None, ), (model_config["in_channels"], None, ), (model_config["in_channels"], None, ), (None,)))
    # net.summary()
    net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    net.fit(
        x=X,
        y=z,
        batch_size=64,
        epochs=1000,
        callbacks=[tensorboard_callback,
                   earlyStop_loss_callback,
                   earlyStop_accu_call_back,
                   best_checkpoint_callback],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/config/config_SSSD.json',
                        help='JSON file for configuration')

    args = parser.parse_args()
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
    train(**train_config)

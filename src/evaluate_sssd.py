import os
import argparse
import json
import numpy as np
from datetime import datetime
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams, get_mask_holiday
from functools import partial
from tensorflow import keras
import tensorflow as tf
from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDImputer import SSSDImputer
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from statistics import mean
from einops import rearrange

def generate(output_directory,
             num_samples,
             ckpt_path,
             masking,
             missing_rate,
             only_generate_missing,
             batch_size,
             alg,
             stock,
             model_loc=None):
    """
    Generate data based on ground truth

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 50
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded;
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """

    # generate experiment (local) path
    past_time = '00000000-000000'
    files_list = os.listdir(output_directory + stock + '/SSSD-' + alg)
    # files_list = os.listdir(output_directory   + '/SSSD-' + alg)
    for file in files_list:
        if file.startswith('2023'):
            past_time = max(past_time, file)
    past_time = model_loc if model_loc is not None else past_time
    local_path = stock + '/SSSD-' + alg + '/' + past_time + '/generated_samples'
    # local_path = 'SSSD-' + alg + '/' + past_time + '/generated_samples'

    # Get shared output_directory ready
    output_directory = output_directory + local_path
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory: ", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key]


    # load checkpoint
    ckpt_path = ckpt_path + stock +'/SSSD-' + alg + '/' + past_time + '/sssd_model'
    # ckpt_path = ckpt_path + 'SSSD-' + alg + '/' + past_time + '/sssd_model'

    try:
        # reload model
        # net = keras.models.load_model(ckpt_path)
        net = SSSDImputer(**model_config, alg=alg)
        print('Successfully loaded model saved at time: {}!'.format(past_time))
    except:
        raise Exception('No valid model found')

    ### Custom data loading and reshaping ###
    data_name = 'scaled_' + stock + '_all_stocks_2013-01-02_to_2023-01-01.npy'
    testing_data = np.load(trainset_config['test_data_path'] + data_name)
    # testing_data = testing_data[:int(testing_data.shape[0] / batch_size) * batch_size]
    testing_data = np.array_split(testing_data, testing_data.shape[0] // batch_size + 1, 0)
    test_gt_masks = np.load('../log/stocks/'+ stock +'/SSSD-'+ alg + '/'+ past_time + '/sssd_log/gt_mask.npy')
    test_gt_masks = np.array_split(test_gt_masks.transpose([0, 2, 1]), test_gt_masks.shape[0] // batch_size + 1, 0)

    print('Data loaded')

    all_mse = []
    all_generated_samples=[]
    pbar = tqdm(total=len(testing_data))
    for i, batch in enumerate(testing_data):
        B, L, C = batch.shape  # B is batchsize, C is the dimension of each audio, L is audio length
        if test_gt_masks is None:
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
            elif masking == 'holiday':
                observed_masks = ~np.isnan(batch)
                holidays = np.unique(np.where(~observed_masks)[0])
                gt_days = holidays

                if holidays.shape == 0:
                    random_day = np.random.choice(np.arange(0, B),
                                                  size=int(np.ceil(B / 16)), replace=False)
                    gt_days = np.append(gt_days, random_day)
                else:
                    random_day = np.random.choice(np.arange(0, B))
                    gt_days = np.append(gt_days, random_day)

                gt_days = np.unique(gt_days)
                mask = observed_masks
                mask[gt_days] = np.zeros_like(observed_masks[0], dtype=bool)
        else:
            observed_masks = (~np.isnan(batch)).astype(float).transpose([0, 2, 1])
            mask = tf.convert_to_tensor(test_gt_masks[i].transpose([0, 2, 1])) # B C L
        mask = tf.cast(mask, tf.float32)
        batch = tf.cast(tf.convert_to_tensor(np.nan_to_num(batch)), tf.float32)
        batch = tf.transpose(batch, perm=[0, 2, 1]) # B C L

        # lmd_generator = lambda i: generator(batch, mask)
        # generated_audio = tf.vectorized_map(lmd_generator, elems=tf.range(num_samples))
        generated_audio = sampling(net=net,
                                   diffusion_hyperparams=diffusion_hyperparams,
                                   num_samples=num_samples,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)

        generated_audio = generated_audio.numpy() # num_sample B L C
        all_generated_samples.append(generated_audio)
        # batch = batch.numpy()
        # mask = mask.numpy()
        #
        # outfile = f'imputation{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, generated_audio)
        #
        # outfile = f'original{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, batch)
        #
        # outfile = f'mask{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, mask)
        #
        # print('saved generated samples at iteration %s' % ckpt_iter)
        generated_audio_median = np.median(generated_audio, axis=0)
        target_mask = observed_masks - mask.numpy()
        if np.sum(target_mask)!= 0:
            mse = mean_squared_error(generated_audio_median[target_mask.astype(bool)], batch[target_mask.astype(bool)])
            mse /= np.sum(target_mask)
        else:
            mse = 0.
        all_mse.append(mse)

        # if i % 5 == 0 and i > 0:
        pbar.update(1)

    print('Total MSE:', mean(all_mse))
    imputed_data = tf.stack(all_generated_samples)
    imputations = rearrange(imputed_data, 'i j b k l -> (i b) j l k') # i: total_timestamps // num_samples, j: num_samples
    imp_data_numpy = imputations.numpy() # B num_samples length feature
    np.save(output_directory + '/imputed_data.npy', imp_data_numpy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/config_SSSD_stocks.json',
                        help='JSON file for configuration')
    parser.add_argument('-n', '--num_samples', type=int, default=25,
                        help='Number of utterances to be generated')
    parser.add_argument('--ignore_warning', type=str, default=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--alg', type=str, default='S4')
    parser.add_argument('--stock', type=str, default='SE')
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

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

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

    generate(**gen_config,
             num_samples=args.num_samples,
             masking=train_config["masking"],
             missing_rate=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"],
             batch_size=train_config['batch_size'],
             alg=args.alg,
             stock=args.stock,
             model_loc=None,
             )

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
    past_time = '00000000-000000'+ '_T_{}_Layers_{}'.format(diffusion_config['T'], model_config['num_res_layers'])
    files_list = os.listdir(output_directory + stock + '/SSSD-' + alg)
    # files_list = os.listdir(output_directory   + '/SSSD-' + alg)
    for file in files_list:
        if file.startswith('2023') and file.endswith('_T_{}_Layers_{}'.format(diffusion_config['T'], model_config['num_res_layers'])):
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
    ckpt_path = ckpt_path + stock +'/SSSD-' + alg + '/' + past_time
    # ckpt_path = ckpt_path + 'SSSD-' + alg + '/' + past_time + '/sssd_model'

    try:
        # reload model
        net = keras.models.load_model(ckpt_path)
        # net = SSSDImputer(**model_config, alg=alg)
        print('Successfully loaded model saved at time: {}!'.format(past_time))
    except:
        raise Exception('No valid model found')

    ### Custom data loading and reshaping ###
        # prepare X and Y for model.fit()
    if stock == 'all':
        train_data = []
        for ticker in ['US', 'EU', 'HK']:
            data_name = 'scaled_' + ticker + '_all_stocks_2018-01-02_to_2023-01-01.npy'
            ticker_data = np.load(trainset_config['train_data_path'] + data_name, allow_pickle=True)[305:]
            train_data.append(ticker_data.astype(np.float32))
        testing_data = np.concatenate(train_data, axis=1)
    else:
        data_name = 'scaled_' + stock + '_all_stocks_2013-01-02_to_2023-01-01.npy'
        testing_data = np.load(trainset_config['train_data_path'] + data_name)  # [1609:]

    L, N, C = testing_data.shape

    testing_data = np.array_split(testing_data.transpose([1, 2, 0]),  N // batch_size + 1, 0) # N C L
    test_gt_masks = np.load('../log/stocks/'+ stock +'/SSSD-'+ alg + '/'+ past_time +'/gt_mask.npy')
    test_gt_masks = np.array_split(test_gt_masks,  N // batch_size + 1, 0)# N C L

    print('Data loaded')

    all_mse = []
    all_generated_samples=[]
    pbar = tqdm(total=len(testing_data))
    for i, batch in enumerate(testing_data):
        observed_masks = (~np.isnan(batch)).astype(float) #.transpose([0, 2, 1])
        mask = tf.convert_to_tensor(test_gt_masks[i])
        mask = tf.cast(mask, tf.float32)
        batch = tf.cast(tf.convert_to_tensor(np.nan_to_num(batch)), tf.float32)


        generated_audio = sampling(net=net,
                                   diffusion_hyperparams=diffusion_hyperparams,
                                   num_samples=num_samples,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)

        generated_audio = generated_audio.numpy() # num_sample N L C
        all_generated_samples.append(generated_audio)
        generated_audio_median = np.median(generated_audio, axis=0)
        target_mask = observed_masks - mask.numpy()
        if np.sum(target_mask)!= 0:
            mse = mean_squared_error(generated_audio_median[target_mask.astype(bool)], batch[target_mask.astype(bool)])
            all_mse.append(mse)
        pbar.update(1)

    print('Total MSE:', mean(all_mse))
    imputed_data = np.concatenate(all_generated_samples, axis=1) # num_samples x 2609 x L x C
    # num_samples B  length feature
    np.save(output_directory + '/imputed_data.npy', imputed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/config_SSSD_stocks.json',
                        help='JSON file for configuration')
    parser.add_argument('-n', '--num_samples', type=int, default=10,
                        help='Number of utterances to be generated')
    parser.add_argument('--ignore_warning', type=str, default=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--algo', type=str, default='S4')
    parser.add_argument('--stock', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=4)
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
             batch_size=args.batch_size,#train_config['batch_size'],
             alg=args.algo,
             stock=args.stock,
             model_loc=None,
             )

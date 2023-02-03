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
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
import sys
from einops import rearrange

def generate(num_samples,
             only_generate_missing,
             batch_size,
             alg,
             stock):
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


    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key]
    net = SSSDImputer(**model_config, alg=alg)
    # load checkpoint
    ckpt_path = "../results/stocks/" + stock +'/SSSD-' + alg + '/' + past_time
    try:
        # reload model
        net.load_weights(ckpt_path).expect_partial()
        # net = SSSDImputer(**model_config, alg=alg)
        print('Successfully loaded model saved at time: {}!'.format(past_time))
    except:
        raise Exception('No valid model found')

    ### Custom data loading and reshaping ###
        # prepare X and Y for model.fit()

    for ticker in ['ES', 'SE', 'DJ']:
        eval_data = []
        eval_mask = []
        print('Evaluating ' + ticker)
        data_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
        mask_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01_gt_masks.npy'
        ticker_data = np.load('../datasets/Stocks/' + data_name, allow_pickle=True).astype(np.float32)
        scalar0 = MinMaxScaler()
        ticker_data = np.array([scalar0.fit_transform(tk) for tk in ticker_data.transpose([1, 0, 2])]).transpose(
            [1, 0, 2])  # N L C -> L N C
        # generate masks: observed_masks + man_made mask
        ticker_mask = np.load(train_log_dir + '/' + mask_name, allow_pickle=True).astype(np.float32)  # L N C
        for i in range(ticker_data.shape[0] // args.seq_len):
            ticker_chunk = ticker_data[args.seq_len * i:args.seq_len * (i + 1)]  # L N C
            eval_data.append(ticker_chunk)
            eval_mask.append(ticker_mask[args.seq_len * i:args.seq_len * (i + 1)])
        eval_data = np.concatenate(eval_data, axis=1).transpose([1, 2, 0]) # L N C -> N L C
        eval_mask = np.concatenate(eval_mask, axis=1).transpose([1, 2, 0]) # L N C -> N L C
        L, N, C = eval_data.shape

        eval_data = np.array_split(eval_data, eval_data.shape[0]//batch_size + 1, 0)
        eval_mask = np.array_split(eval_mask, eval_mask.shape[0]//batch_size + 1, 0)

        print('Data loaded')

        ###detect generation in output_directory
        # output_directory = output_directory + ticker
        if not os.path.exists(output_directory + '/' + ticker):
            os.mkdir(output_directory + '/' + ticker)
        generate_lists = os.listdir(output_directory + '/' + ticker)
        all_generated_samples = []
        generated_num = 0
        for file in generate_lists:
            if file.startswith('imputed_batch_') and file.endswith("_data.npy"):
                generated_audio = np.load(output_directory + '/' + ticker + '/' + file, allow_pickle=True).astype(np.float32)
                all_generated_samples.append(generated_audio)
                generated_num += 1
        print('Have {} samples been generated, start from {}'.format(generated_num, generated_num+1))
        eval_data = eval_data[generated_num:]
        eval_mask = eval_mask[generated_num:]
        all_mse = []
        all_mae = []

        pbar = tqdm(total=len(eval_data))
        for i, batch in enumerate(eval_data):
            observed_masks = (~np.isnan(batch)).astype(float) #.transpose([0, 2, 1])
            mask = tf.convert_to_tensor(eval_mask[i])
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
                mae = np.mean(abs(generated_audio_median[target_mask.astype(bool)] - batch[target_mask.astype(bool)]))
                all_mse.append(mse)
                all_mae.append(mae)
                print('Current batch {} MSE is {} MAE is {}'.format(i, mse, mae))
            pbar.update(1)
            np.save(output_directory  + '/' + ticker +'/imputed_batch_'+str(i)+'_data.npy', generated_audio) # num_sample N L C

        print('Total MSE:', mean(all_mse))
        print("Total MAE", mean(all_mae))
        imputed_data = np.concatenate(all_generated_samples, axis=1) # num_samples x N x L x C
        # num_samples B  length feature
        np.save(output_directory + '/' + ticker + '/imputed_all_data.npy', imputed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', default=200)
    parser.add_argument('--seq_len', default=100, type=int)
    parser.add_argument('--num_layers', default=36)
    parser.add_argument('-n', '--num_samples', type=int, default=10,
                        help='Number of utterances to be generated')
    parser.add_argument('--ignore_warning', type=str, default=True)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--alg', type=str, default='S4')
    parser.add_argument('--stock', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=128)
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

    # generate experiment (local) path
    past_time = '00000000000000' #+ '_seq_{}_T_{}_Layers_{}'.format(args.seq, args.T, 20)
    files_list = os.listdir("../results/stocks/" + args.stock + '/SSSD-' + args.alg)
    # files_list = os.listdir(output_directory   + '/SSSD-' + alg)
    for file in files_list:
        if file.startswith('202302') and file.endswith('_seq_{}_T_{}_Layers_{}'.format(args.seq_len, args.T, args.num_layers)):
            past_time = max(int(past_time), int(file[:8] + file[9:15]))
    past_time = str(past_time)[:8] + '-' + str(past_time)[8:] + '_seq_{}_T_{}_Layers_{}'.format(args.seq_len, args.T, args.num_layers)
    # past_time = '20230201-134917_seq_100_T_100_Layers_20'
    local_path = args.stock + '/SSSD-' + args.alg + '/' + past_time + '/generated_samples'
    # local_path = 'SSSD-' + alg + '/' + past_time + '/generated_samples'

    # Get shared output_directory ready
    output_directory = "../results/stocks/" + local_path
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory: ", output_directory, flush=True)

    train_log_dir = '../log/stocks/' + args.stock + '/SSSD-' + args.alg + '/' + past_time +'/'
    config_name = 'config_SSSD_stocks_seq_{}_T_{}_Layers_{}.json'.format(args.seq_len, args.T, args.num_layers)


    # Parse configs. Globals nicer in this case
    sys.path.append(train_log_dir + config_name)
    with open(train_log_dir + config_name) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    config = json.loads(data)

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

    generate(num_samples=args.num_samples,
             only_generate_missing=train_config["only_generate_missing"],
             batch_size=args.batch_size,
             alg=args.alg,
             stock=args.stock,
             )

from imputers.CSDIImputer_1 import *
from imputers.CSDIImputer import CDSIimputer
import os
import argparse
from einops import rearrange
from sklearn.preprocessing import MinMaxScaler
import sys
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_loc', type=str, default=None, help='The location of the log file')
    parser.add_argument('--alg', type=str, default='S4', help='The Algorithm for imputation: transformer or S4')
    parser.add_argument('--data', type=str, default='stocks', help='The data set for training')
    parser.add_argument('--stock', type=str, default='all', help='The data set for training: DJ SE ES')
    parser.add_argument('--cuda', type=int, default=0, help='The CUDA device for training')
    parser.add_argument('--n_samples', '-n', type=int, default=10, help='The number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of batch size')
    parser.add_argument('--masking', type=str, default='holiday', help='The masking strategy')
    parser.add_argument('--target_strategy', type=str, default='holiday', help='The target strategy')
    parser.add_argument('--amsgrad', type=bool, default=False, help='The optimizer whether uses AMSGrad')
    parser.add_argument('--seq_len', default=800, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
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
    model_path = '../results/' + args.data + '/' + args.stock + '/CSDI-' + args.alg + '/'
    # generate experiment (local) path
    past_time = '00000000000000'  # + '_seq_{}_T_{}_Layers_{}'.format(args.seq, args.T, 20)
    files_list = os.listdir("../results/stocks/" + args.stock + '/CSDI-' + args.alg)
    # files_list = os.listdir(output_directory   + '/SSSD-' + alg)
    for file in files_list:
        if file.startswith('202302') and file.endswith('_seq_{}'.format(args.seq_len)):
            past_time = max(int(past_time), int(file[:8] + file[9:15]))
    target_file = str(past_time)[:8] + '-' + str(past_time)[8:] + '_seq_{}'.format(args.seq_len)
    # past_time = '20230201-134917_seq_100_T_100_Layers_20'
    generated_path = model_path + target_file + '/generated_samples'
    log_path = '../log/' + args.data + '/' + args.stock +'/CSDI-' + args.alg + '/'
    log_path = log_path + target_file
    model_path = model_path + target_file
    config_path = './config/' #+ "/config_csdi_training_" + args.masking
    # load data from training
    config_filename = log_path + "/config_csdi_training_holiday.json"
    # Parse configs. Globals nicer in this case
    sys.path.append(config_filename)
    with open(config_filename) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    config = json.loads(data)

    CSDI = CSDI(model_path,
                              log_path,
                              config_path,
                              masking=args.masking,
                              algo=args.alg,
                              batch_size=args.batch_size,
                              target_strategy=args.target_strategy,
                              amsgrad=args.amsgrad,
                              training=False)
    CSDI.model = CDSIimputer(5, config)
    CSDI.model.load_weights(model_path).expect_partial()
    ## Custom data loading and reshaping ###
    # prepare X and Y for model.fit()

    for ticker in ['ES', 'SE', 'DJ']:
        eval_data = []
        eval_mask = []
        # ticker = 'SE'
        print('Evaluating '+ ticker)
        data_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
        mask_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01_gt_masks.npy'
        ticker_data = np.load('../datasets/Stocks/' + data_name, allow_pickle=True).astype(np.float32)
        scalar0 = MinMaxScaler()
        ticker_data = np.array([scalar0.fit_transform(tk) for tk in ticker_data.transpose([1, 0, 2])]).transpose(
            [1, 0, 2])  # N L C -> L N C
        # generate masks: observed_masks + man_made mask
        ticker_mask = np.load(log_path + '/' + mask_name, allow_pickle=True).astype(np.float32)  # L N C
        for i in range(ticker_data.shape[0] // args.seq_len):
            ticker_chunk = ticker_data[args.seq_len * i:args.seq_len * (i + 1)]  # L N C
            eval_data.append(ticker_chunk)
            eval_mask.append(ticker_mask[args.seq_len * i:args.seq_len * (i + 1)])
        eval_data = np.concatenate(eval_data, axis=1).transpose([1, 2, 0])  # L N C -> N L C
        eval_mask = np.concatenate(eval_mask, axis=1).transpose([1, 2, 0])  # L N C -> N L C
        print("Loaded Data from:{}".format(target_file))
        # training_data = rearrange(tf.convert_to_tensor(observed_data[:16]), 'b l k -> b k l')
        imputed_data = CSDI.imputer(sample=eval_data, gt_mask=eval_mask, n_samples=args.n_samples, ticker=ticker)
        # imputations = imputed_data.stack()  # int(sample.shape[0]/self.batch_size) * n_samples * B * L * K
        np.save(generated_path + '/' + ticker +'/imputed_all_data.npy', imputed_data)



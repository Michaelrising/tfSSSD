import numpy as np

from imputers.CSDIImputer_1 import *
import argparse
from datetime import datetime
from utils.util import get_mask_holiday
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='S5', help='The Algorithm for imputation: transformer or S4')
    parser.add_argument('--data', type=str, default='stocks', help='The data set for training')
    parser.add_argument('--stock', type=str, default='all', help='The data set for training: DJ SE ES')
    parser.add_argument('--cuda', type=int, default=1, help='The CUDA device for training')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='The number of batch size')
    parser.add_argument('--masking', type=str, default='holiday', help='The masking strategy')
    parser.add_argument('--target_strategy', type=str, default='holiday', help='The target strategy')
    parser.add_argument('--amsgrad', type=bool, default=False, help='The optimizer whether uses AMSGrad')
    parser.add_argument('--seq_len', type=int, default=200)
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
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = '../results/' + args.data + '/' + args.stock +'/CSDI-' + args.algo + '/' + current_time + '_seq_{}'.format(args.seq_len)# + '/'
    log_path = '../log/' + args.data + '/' + args.stock + '/CSDI-' + args.algo + '/' + current_time + '_seq_{}'.format(args.seq_len)
    config_path = './config'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    if args.data == 'mujoco':
        all_data = np.load('../datasets/Mujoco/train_mujoco.npy')
        all_data = np.array(all_data)
        training_data = tf.convert_to_tensor(all_data[:7680]) # B L K
        # validation_data = tf.convert_to_tensor(all_data[6400:7680])
        predicton_data = tf.convert_to_tensor(all_data[7680:])
    if args.data == 'stocks':
        # Stock data
        training_mask = []
        training_data = []
        for ticker in ['DJ', 'ES', 'SE']:
            data_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
            ticker_data = np.load('../datasets/Stocks/' + data_name, allow_pickle=True).astype(np.float32)
            scalar0 = MinMaxScaler()
            ticker_data = np.array([scalar0.fit_transform(tk) for tk in ticker_data.transpose([1, 0, 2])]).transpose([1, 0, 2])  # N L C -> L N C
            # generate masks: observed_masks + man_made mask
            ticker_mask = get_mask_holiday(ticker_data)  # N L C
            ticker_mask = ticker_mask.numpy().transpose([1, 0, 2]) # L N C
            # for i in range(0, ticker_data.shape[0] - args.seq_len, args.mw):
            #     ticker_chunk = ticker_data[i:i + args.seq_len]  # L N C
            #     training_data.append(ticker_chunk)
            #     training_mask.append(ticker_mask[i:i + args.seq_len])
            for i in range(ticker_data.shape[0] // args.seq_len):
                ticker_chunk = ticker_data[args.seq_len * i:args.seq_len * (i + 1)]  # L N C
                training_data.append(ticker_chunk)
                training_mask.append(ticker_mask[args.seq_len * i:args.seq_len * (i + 1)])
            np.save(log_path + '/' + ticker + '_all_stocks_2013-01-02_to_2023-01-01_gt_masks.npy', ticker_mask)
        training_data = np.concatenate(training_data, axis=1).astype(float)  # L B K
        training_mask = np.concatenate(training_mask, axis=1).astype(float)  # L B K
        # shuffle data
        training_all = np.concatenate((training_data.transpose([1, 0, 2]), training_mask.transpose([1, 0, 2])), axis=-1)
        np.random.shuffle(training_all)
        training_data = training_all[..., :5].transpose([1,0,2]) # L B K
        training_mask = training_all[..., 5:].transpose([1,0,2]) # L B K
        print(training_data.shape)
    print('Data loaded')
    CSDIImputer = CSDI(model_path,
                              log_path,
                              config_path,
                              masking=args.masking,
                              epochs=args.epochs,
                              algo=args.algo,
                              batch_size=args.batch_size,
                              target_strategy=args.target_strategy,
                              amsgrad=args.amsgrad
                              )
    X = (training_data, training_mask) # L B K
    CSDIImputer.train(X, masking=args.masking)




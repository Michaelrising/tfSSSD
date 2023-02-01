import pandas as pd

from predictors.Predictor import Predictor
import argparse
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def evaluate(model, test_data):
    X= test_data
    Y = model(X)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1, help='The CUDA device for training')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of batch size')
    parser.add_argument('--amsgrad', type=bool, default=False, help='The optimizer whether uses AMSGrad')
    parser.add_argument('--win', type=float, default=1.) # 1 year window
    parser.add_argument('-pre_len', type=int, default=30) # prediction length
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
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    Model_path = '../results/prediction_task/' + current_time + '/'
    Log_path = '../log/prediction_task/' + current_time + '/'

    # has nan
    imputer = KNNImputer(n_neighbors=2, weights="uniform")

    def simple_imputer(x):
        masks = np.isnan(x[:, 0])
        index = np.where(masks)[0]
        imputation = []
        for d in index:
            choice = np.array([d-4,d-3, d-2, d-1, d+1, d+2, d+3, d+4, d+5])
            choice = choice[(choice > 0)*(choice<x.shape[0]-1)]
            m = ~np.isin(choice, index)
            choice = choice[m]
            dd = x[choice].reshape(-1, x.shape[1])
            imputation.append(np.mean(dd, axis=0))
        x[masks] = np.array(imputation)
        return x


    US = np.load('../datasets/Stocks/scaled_DJ_all_stocks_2013-01-02_to_2023-01-01.npy', allow_pickle=True).transpose([1, 0, 2]) # N L C
    US = np.array([simple_imputer(us.astype(float)) for us in US])
    EU = np.load('../datasets/Stocks/scaled_ES_all_stocks_2013-01-02_to_2023-01-01.npy', allow_pickle=True).transpose([1, 0, 2]) # N L C
    EU = np.array([simple_imputer(eu.astype(float)) for eu in EU])
    EU = np.array([imputer.fit_transform(eu.astype(float)) for eu in EU])
    HK = np.load('../datasets/Stocks/scaled_SE_all_stocks_2013-01-02_to_2023-01-01.npy', allow_pickle=True).transpose([1, 0, 2]) # N L C
    HK = np.array([simple_imputer(hk.astype(float)) for hk in HK])
    X = np.concatenate((US[..., :-1], EU[..., :-1], HK[..., :-1]), axis=0).transpose([1, 0, 2])  # L N1+N2+N3 C
    # X_diff = X[..., 1] - X[..., 2] # L N1+N2+N3
    # X = X_diff
    win = int(args.win * 365 - 52 * 2)
    assert win < US.shape[1] / 2

    Train_X = X[:-args.pre_len]
    Test_X = X[-(win + args.pre_len):]


    train_x = []
    for i in range(0, Train_X.shape[0] - win):
        train_x.append(Train_X[i:(i + win)])

    test_x = []
    for i in range(0, Test_X.shape[0] - win):
        test_x.append(Test_X[i:(i + win)])
    train_x = np.stack(train_x)  # B W N'
    test_x = np.stack(test_x).astype(np.float32)
    for file in os.listdir('../datasets/Stocks/SE'):
        print('Start Training Stock: ' + file[:7])
        HK = pd.read_csv('../datasets/Stocks/SE/'+file)
        HK = HK[['Date.1', 'High', 'Low', 'Close', 'Adj Close']].to_numpy()  # L C
        HK = simple_imputer(HK)  # L C=1
        HK_diff = (HK[:, 1] - HK[:, 2]).reshape(-1, 1)
        scalar = MinMaxScaler()
        scalar.fit(HK_diff)
        HK_diff = scalar.transform(HK_diff)

        # Only predict the difference between high and low for HK stock L * N3
        train_y = HK_diff[win:-args.pre_len]
        pred_y = HK_diff[-args.pre_len:]
        data = (train_x, train_y)
        model_path = Model_path + file[:7] + '/'
        log_path = Log_path + file[:7] + '/'
        predictor = Predictor(
            model_path,
            log_path,
            features=256,
            mid_feature=64,
            out_feature=train_y.shape[1],
            depth=4,
            chunk_size=-1,
        )
        predictor.train(data,
                        epochs=10,
                        batch_size=args.batch_size)
        print('Finish Training Stock: ' + file[:7])
        print('Start Predicting Stock: ' + file[:7])
        prediction = predictor.evaluate(test_x)
        maxi = scalar.data_max_
        mini = scalar.data_min_
        prediction = prediction*(maxi - mini) + mini
        pred_y = pred_y*(maxi - mini) + mini
        mse = np.mean((pred_y - prediction)**2)
        print('Finish Predicting Stock: ' + file[:7]+' With MSE: '+str(mse))
        plt.plot(pred_y, c='blue', label='True', ls='-', lw=1.5)
        plt.plot(prediction, c='red', label='Pred', ls='-.', lw=1.5)
        plt.legend()
        plt.savefig(log_path + file[:7] + '_visualization.png')
        plt.close()











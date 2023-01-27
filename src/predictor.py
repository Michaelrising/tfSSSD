from imputers.MegaPredictor import MegaPredictor
import argparse
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.impute import KNNImputer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1, help='The CUDA device for training')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of batch size')
    parser.add_argument('--amsgrad', type=bool, default=False, help='The optimizer whether uses AMSGrad')
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

    model_path = '../results/prediction_task/' + current_time
    log_path = '../log/prediction_task/' + current_time
    predictor = MegaPredictor(
                             model_path,
                             log_path,
                             features=128,
                             depth=8,
                             chunk_size=16,

                            )
    # has nan
    imputer = KNNImputer(n_neighbors=6, weights="uniform")
    dj30 = imputer.fit_transform(np.load('../datasets/Stocks/scaled_DJ_all_stocks_2013-01-02_to_2023-01-01.npy').reshape(2609, -1)).reshape(-1, 2609, 6)
    es50 = imputer.fit_transform(np.load('../datasets/Stocks/scaled_ES_all_stocks_2013-01-02_to_2023-01-01.npy').reshape(2609, -1)).reshape(-1, 2609, 6)
    hs70 = imputer.fit_transform(np.load('../datasets/Stocks/scaled_SE_all_stocks_2013-01-02_to_2023-01-01.npy').reshape(2609, -1)).reshape(-1, 2609, 6)
    data = (dj30[:, -1], es50[:, :-1], hs70[:, 1:])

    predictor.train(data)





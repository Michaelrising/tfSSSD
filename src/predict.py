from predictors.Predictor import Predictor
import argparse
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

def evaluate(model, test_data):
    X= test_data
    Y = model(X)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1, help='The CUDA device for training')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of batch size')
    parser.add_argument('--amsgrad', type=bool, default=True, help='The optimizer whether uses AMSGrad')
    parser.add_argument('--seq_len', type=int, default=200) # 1 year window
    parser.add_argument('--model', type=str, default='sssd-s4', help='mega or sssd-s4')
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
    model_name = args.model.upper() + '/' + current_time + '/'
    model_path = '../results/prediction_task/' + model_name +'/'
    log_path = '../log/prediction_task/' + model_name +'/'
    data_path = '../generation/stocks/CSDI-S4/20230202-105732_seq_len_200'
    predictor = Predictor(
        model_path,
        log_path,
        in_channels=5,
        out_channels=5,
        model=args.model
            )
    predictor.train(data_path,
                    epochs=50,
                    batch_size=args.batch_size)

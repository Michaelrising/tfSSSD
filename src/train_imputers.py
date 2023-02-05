import pandas as pd

from imputers.Imputers import Imputer
import argparse
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore_warning', type=bool, default=True)
    parser.add_argument('--cuda', type=int, default=0, help='The CUDA device for training')
    parser.add_argument('--epoch', type=int, default=1, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of batch size')
    parser.add_argument('--amsgrad', type=bool, default=False, help='The optimizer whether uses AMSGrad')
    parser.add_argument('--seq_len', type=int, default=200) # 1 year window
    parser.add_argument('--model', type=str, default='mega', help='The model used for imputation lower class, sssd csdi or mega')
    parser.add_argument('--alg', type=str, default=None, help='The algorithm used in sssd or csdi None for mega')
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

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name = args.model.upper() + ("-" + args.alg) if args.alg is not None else args.model.upper()
    model_path = '../results/stocks/all/'+model_name
    log_path = '../log/stocks/all/'+model_name
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    imputer = Imputer(
                    model_path,
                    log_path,
                    model=args.model,
                    alg=args.alg,
                    seq_len=args.seq_len,
                    in_channels=5,
                    out_channels=5)
    imputer.train(
                  epoch=args.epoch,
                  batch_size=args.batch_size)

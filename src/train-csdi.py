import numpy as np

from imputers.CSDIImputer import *
import argparse
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='S4', help='The Algorithm for imputation: transformer or S4')
    parser.add_argument('--data', type=str, default='stocks', help='The data set for training')
    parser.add_argument('--stock', type=str, default='all', help='The data set for training: DJ SE ES')
    parser.add_argument('--cuda', type=int, default=1, help='The CUDA device for training')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of batch size')
    parser.add_argument('--masking', type=str, default='holiday', help='The masking strategy')
    parser.add_argument('--target_strategy', type=str, default='holiday', help='The target strategy')
    parser.add_argument('--amsgrad', type=bool, default=False, help='The optimizer whether uses AMSGrad')
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--mw', default=5, type=int)
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
    model_path = '../results/' + args.data + '/' + args.stock +'/CSDI-' + args.algo + '/' + current_time + '_seq_{}'.format(args.seq_len)
    log_path = '../log/' + args.data + '/' + args.stock + '/CSDI-' + args.algo + '/' + current_time + '_seq_'.format(args.seq_len)
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
        if args.stock == 'all':
            # prepare X and Y for model.fit()
            train_data = []
            for ticker in ['US', 'HK', 'EU']:
                data_name = 'scaled_' + ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
                ticker_data = np.load('../datasets/Stocks/' + data_name, allow_pickle=True)
                for i in range(ticker_data.shape[0] // args.seq_len):
                    train_data.append(ticker_data[args.seq_len * i:args.seq_len * (i + 1)])
            train_data = np.concatenate(train_data, axis=1).astype(float) # L B K
            train_data = train_data.transpose([1,0,2]) # B L K
            np.random.shuffle(train_data)

            print('Loading stocks data: ' + args.stock)
            print(train_data.shape) # L B K
        elif args.stock == 'single':
            train_data = []
            data_name = 'scaled_DJ_all_stocks_2013-01-02_to_2023-01-01.npy'
            ticker_data = np.load('../datasets/Stocks/' + data_name, allow_pickle=True) #.transpose([1, 0, 2])
            ticker_data = ticker_data[:, 0]# L * K
            for i in range(0, ticker_data.shape[0]-args.seq_len, args.mw):
                train_data.append(ticker_data[i:args.seq_len + i]) # Seq_len * 6
            train_data = np.stack(train_data).astype(float) # B L K
            # train_data = train_data.transpose([1,0,2])
        else:
            train_data = np.load('../datasets/Stocks/scaled_' + args.stock +'_all_stocks_2013-01-02_to_2023-01-01.npy') # Time_length * num_stocks * feature [B L K]
        training_data = tf.convert_to_tensor(train_data)
        # validation_data = tf.convert_to_tensor(all_data[int(0.8*all_data.shape[0]):])
    print('Data loaded')
    CSDIImputer = CSDIImputer(model_path,
                              log_path,
                              config_path,
                              masking=args.masking,
                              epochs=args.epochs,
                              algo=args.algo,
                              batch_size=args.batch_size,
                              target_strategy=args.target_strategy,
                              amsgrad=args.amsgrad
                              )
    train_data = CSDIImputer.train(training_data, masking=args.masking)
    # test_data = tf.convert_to_tensor(training_data[7000:])
    observed_data, ob_mask, gt_mask, _ = train_data

    observed_data = observed_data.numpy()
    ob_mask = ob_mask.numpy()
    gt_mask = gt_mask.numpy()
    np.save(log_path + '/observed_data.npy', observed_data)
    np.save(log_path + '/ob_mask.npy', ob_mask)
    np.save(log_path + '/gt_mask.npy', gt_mask)

    # imputed_data = CSDIImputer.imputer(sample=observed_data, gt_mask=gt_mask, ob_masks=ob_mask, n_samples=50)
    # # imputations = imputed_data.stack()  # int(sample.shape[0]/self.batch_size) * n_samples * B * K * L
    # imputations = rearrange(imputed_data, 'i j b k l -> i b j l k')
    # # ob_data_numpy = observed_data
    # # gt_mask_numpy = gt_mask
    # # # indx_imputation = ~gt_mask_numpy
    # imp_data_numpy = imputations.numpy()
    # np.save(log_path+'/imputed_data.npy', imp_data_numpy)
    #

    # model_imputer = tfCSDI(observed_data.shape[1], CSDIImputer.config, CSDIImputer.device)
    #
    # model_imputer.load_weights(CSDIImputer.model_path).expect_partial()
    #
    # imputed_data = CSDIImputer.imputer(device=device, model=model_imputer, sample=observed_data, gt_mask=gt_mask, ob_masks=ob_mask)
    # # imputations = imputed_data.stack()  # int(sample.shape[0]/self.batch_size) * n_samples * B * K * L
    # imputations = rearrange(imputed_data, 'i j b k l -> i b j l k')
    # ob_data_numpy = observed_data.numpy()
    # gt_mask_numpy = gt_mask.numpy()
    # indx_imputation = ~gt_mask_numpy
    # imp_data_numpy = imputations.numpy()
    #
    # original_sample_replaced = []
    # for original_sample, single_n_samples in zip(ob_data_numpy, imp_data_numpy):  # [x,x,x] -> [x,x] & [x,x,x,x] -> [x,x,x]
    #     single_sample_replaced = []
    #     for sample_generated in single_n_samples:  # [x,x] & [x,x,x] -> [x,x]
    #         j = 0
    #         sample_out = original_sample.copy()
    #         sample_out[indx_imputation] = sample_generated[indx_imputation]
    #         single_sample_replaced.append(sample_out)
    #         j += 1
    #     original_sample_replaced.append(single_sample_replaced)
    # original_sample_replaced = np.array(original_sample_replaced)
    # output = original_sample_replaced



from imputers.CSDIImputer import *
import os
import argparse
from einops import rearrange


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_loc', type=str, default=None, help='The location of the log file')
    parser.add_argument('--algo', type=str, default='S4', help='The Algorithm for imputation: transformer or S4')
    parser.add_argument('--data', type=str, default='stocks', help='The data set for training')
    parser.add_argument('--stock', type=str, default='DJ', help='The data set for training: DJ SE ES')
    parser.add_argument('--cuda', type=int, default=0, help='The CUDA device for training')
    parser.add_argument('--n_samples', '-n', type=int, default=50, help='The number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of batch size')
    parser.add_argument('--masking', type=str, default='holiday', help='The masking strategy')
    parser.add_argument('--target_strategy', type=str, default='holiday', help='The target strategy')
    parser.add_argument('--amsgrad', type=bool, default=False, help='The optimizer whether uses AMSGrad')
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
    model_path = '../results/' + args.data + '/' + args.stock + '/CSDI-' + args.algo + '/'
    files_list = os.listdir(model_path)
    target_file = '00000000-000000'
    for file in files_list:
        if file.startswith('2023'):
            target_file = max(target_file, file)
    target_file = args.model_loc if args.model_loc is not None else target_file
    model_path = model_path + target_file + '/csdi_model'
    log_path = '../log/' + args.data + '/' + args.stock +'/CSDI-' + args.algo + '/'
    log_path = log_path + target_file + '/csdi_log'
    config_path = './config/' #+ "/config_csdi_training_" + args.masking
    # load data from training
    observed_data, ob_mask, gt_mask = np.load(log_path + '/observed_data.npy'), np.load(log_path + '/ob_mask.npy'), np.load(log_path + '/gt_mask.npy')
    print("Loaded Data from:{}".format(target_file))
    # training_data = rearrange(tf.convert_to_tensor(observed_data[:16]), 'b l k -> b k l')
    CSDIImputer = CSDIImputer(model_path,
                              log_path,
                              config_path,
                              masking=args.masking,
                              algo=args.algo,
                              batch_size=args.batch_size,
                              target_strategy=args.target_strategy,
                              amsgrad=args.amsgrad,
                              training=False)

    CSDIImputer.model = keras.models.load_model(model_path)
    imputed_data = CSDIImputer.imputer(sample=observed_data, gt_mask=gt_mask, ob_masks=ob_mask, n_samples=50)
    # imputations = imputed_data.stack()  # int(sample.shape[0]/self.batch_size) * n_samples * B * K * L
    imputations = rearrange(imputed_data, 'i j b k l -> (i b) j l k') # B num_samples length feature
    imp_data_numpy = imputations.numpy()
    np.save(log_path+'/imputed_data.npy', imp_data_numpy)



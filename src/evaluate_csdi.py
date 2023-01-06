from imputers.CSDIImputer import *
import os
import argparse
from einops import rearrange


if __name__ == "__main__":

    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_loc', type=str, default='20230105-151317', help='The location of the log file')
    parser.add_argument('--algo', type=str, default='transformer', help='The Algorithm for imputation: transformer or S4')
    parser.add_argument('--data', type=str, default='mujoco', help='The data set for training')
    parser.add_argument('--cuda', type=int, default=0, help='The CUDA device for training')
    parser.add_argument('--n_samples', '-n', type=int, default=50, help='The number of samples to evaluate')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    model_path = '../results/' + args.data + '/CSDI-' + args.algo + '/'
    files_list = os.listdir(model_path)
    target_file = '00000000-000000'
    for file in files_list:
        if file.startswith('2022'):
            target_file = max(target_file, file)
    target_file = args.model_loc
    model_path = model_path + target_file + '/csdi_model'
    log_path = '../log/' + args.data + '/CSDI-' + args.algo + '/'
    log_path = log_path + target_file + '/csdi_log'
    config_path = './config'
    # load data from training
    observed_data, ob_mask, gt_mask = np.load(log_path + '/observed_data.npy'), np.load(log_path + '/ob_mask.npy'), np.load(log_path + '/gt_mask.npy')
    training_data = rearrange(tf.convert_to_tensor(observed_data[:16]), 'b l k -> b k l')
    CSDIImputer = CSDIImputer(model_path, log_path, config_path, algo=args.algo)
    _, _ = CSDIImputer.train(training_data, infer_flag=True)

    CSDIImputer.model.load_weights(CSDIImputer.model_path)#.expect_partial()

    imputed_data = CSDIImputer.imputer(sample=observed_data, gt_mask=gt_mask, ob_masks=ob_mask, n_samples=5)
    # imputations = imputed_data.stack()  # int(sample.shape[0]/self.batch_size) * n_samples * B * K * L
    imputations = rearrange(imputed_data, 'i j b k l -> i b j l k')
    # ob_data_numpy = observed_data
    # gt_mask_numpy = gt_mask
    # # indx_imputation = ~gt_mask_numpy
    imp_data_numpy = imputations.numpy()
    np.save(log_path+'/imputed_data.npy', imp_data_numpy)

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



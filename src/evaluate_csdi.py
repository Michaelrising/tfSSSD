from imputers.CSDIImputer import *
import os

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = '/gpu:0'
    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = '../results/mujoco/CSDI/'
    files_list = os.listdir(model_path)
    target_file = '00000000-000000'
    for file in files_list:
        if file.startswith('2022'):
            target_file = max(target_file, file)
    model_path = model_path +target_file + '/csdi_model'
    log_path = '../log/mujoco/CSDI/'
    log_path = log_path + target_file + '/csdi_log'
    config_path = './config'
    # load data from training
    observed_data, ob_mask, gt_mask = np.load(log_path + '/observed_data.npy'), np.load(log_path + '/ob_mask.npy'), np.load(log_path + '/gt_mask.npy')

    CSDIImputer = CSDIImputer(model_path, log_path, config_path, time_layer='transformer')

    model_imputer = tfCSDI(observed_data.shape[1], CSDIImputer.config, CSDIImputer.device)

    model_imputer.load_weights(CSDIImputer.model_path).expect_partial()
    CSDIImputer.model = model_imputer

    imputed_data = CSDIImputer.imputer(sample=observed_data, gt_mask=gt_mask, ob_masks=ob_mask)
    # imputations = imputed_data.stack()  # int(sample.shape[0]/self.batch_size) * n_samples * B * K * L
    imputations = rearrange(imputed_data, 'i j b k l -> i b j l k')
    ob_data_numpy = observed_data
    gt_mask_numpy = gt_mask
    indx_imputation = ~gt_mask_numpy
    imp_data_numpy = imputations.numpy()
    np.save(log_path+'/imputed_data.npy')

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



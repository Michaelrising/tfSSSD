import numpy as np
from einops import rearrange, einsum
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import os
model = 'MEGA'
save_path = '../../generation/stocks/MEGA/20230204-213934_seq_800/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
train_log_dir = '../../log/stocks/all/MEGA/20230204-213934_seq_800/'
output_directory = '../../results/stocks/all/MEGA/20230204-213934_seq_800/generated_samples'
seq_len = 800
scaled_all_mae = []
scaled_all_mse = []
origin_all_mae = []
origin_all_mse = []
for ticker in ['ES', 'SE', 'DJ']:
    print('Generating ' + ticker)
    # ticker = 'SE'
    data_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
    mask_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01_gt_masks.npy'
    ticker_data = np.load('../../datasets/Stocks/' + data_name, allow_pickle=True).astype(np.float32)
    scalar0 = MinMaxScaler()
    scaled_ticker_data = np.array([scalar0.fit_transform(tk) for tk in ticker_data.transpose([1, 0, 2])]).transpose([1, 0, 2])  # N L C -> L N C
    # generate masks: observed_masks + man_made mask
    ticker_mask = np.load(train_log_dir + mask_name, allow_pickle=True).astype(np.float32)  # L N C
    generated_data = []
    imputed_data = np.load(output_directory + '/' + ticker + '/imputed_all_data.npy').transpose([1,0,2,3]) # B num_samples seq_len K
    chunks = ticker_data.shape[0] // seq_len
    length = chunks * seq_len
    num_tickers = ticker_data.shape[1]
    for i in range(chunks):
        data = imputed_data[i*num_tickers:(i+1)*num_tickers]
        # data = rearrange(data, 'b n l k -> n (b l ) k', l=seq_len) # num_samples L=2400 K
        generated_data.append(data) # B N L K
    if model == 'SSSD' or 'MEGA':
        generated_data = np.concatenate(generated_data, axis=-1)  # B N L K L=2400
        generated_data = rearrange(generated_data, ' b n k l -> b n l k')
    else:
        generated_data = np.concatenate(generated_data, axis=2) # B N L K L=2400
    generated_data = rearrange(generated_data, ' b n l k -> n l b k')
    scaled_ticker_data = scaled_ticker_data[:length]
    ticker_data = ticker_data[:length]
    ticker_mask = ticker_mask[:length]
    observed_mask = ~np.isnan(ticker_data)
    loss_mask = observed_mask.astype(np.float32) - ticker_mask
    ###### calculate metrics #######
    scaled_diff = generated_data[:,loss_mask.astype(bool)] - scaled_ticker_data[loss_mask.astype(bool)]
    mae = abs(scaled_diff).reshape(-1)
    scaled_all_mae.append(mae)
    mse = (scaled_diff**2).reshape(-1)
    scaled_all_mse.append(mse)
    ##### transform back #####

    mini = np.min(np.nan_to_num(ticker_data), axis=0) # B C
    maxi = np.max(np.nan_to_num(ticker_data), axis=0) # B C
    ranging = maxi - mini
    generated_origin_data = einsum(generated_data, ranging, 'n l b k, b k -> n l b k') + mini
    diff = generated_origin_data[:, loss_mask.astype(bool)] - ticker_data[loss_mask.astype(bool)]
    mae = abs(diff).reshape(-1)
    origin_all_mae.append(mae)
    mse = (diff ** 2 ).reshape(-1)
    origin_all_mse.append(mse)

    scaled_ticker_data[~observed_mask] = np.median(generated_data, axis=0)[~observed_mask].copy()
    ticker_data[~observed_mask] = np.median(generated_origin_data, axis=0)[~observed_mask].copy()
    np.save(save_path+'generated_scaled_'+ticker+'_all_stocks_2013-01-02_to_2023-01-01.npy', scaled_ticker_data)
    np.save(save_path + 'generated_'+ data_name, ticker_data)

scaled_all_mse = np.concatenate(scaled_all_mse).reshape(-1)
mean_scaled_mse = np.mean(scaled_all_mse)
std_scaled_mse = np.std(scaled_all_mse)
lower_limit, upper_limit = mean_scaled_mse - norm.ppf(0.95) * std_scaled_mse/np.sqrt(scaled_all_mse.shape[0]), mean_scaled_mse + norm.ppf(0.95) * std_scaled_mse/np.sqrt(scaled_all_mse.shape[0])
print("Scaled mse:{}, ({}, {})".format(mean_scaled_mse, lower_limit, upper_limit))

scaled_all_mae = np.concatenate(scaled_all_mae).reshape(-1)
mean_scaled_mae = np.mean(scaled_all_mae)
std_scaled_mae = np.std(scaled_all_mae)
lower_limit, upper_limit = mean_scaled_mae - norm.ppf(0.95) * std_scaled_mae/np.sqrt(scaled_all_mae.shape[0]), mean_scaled_mae + norm.ppf(0.95) * std_scaled_mae/np.sqrt(scaled_all_mae.shape[0])
print("Scaled mae:{}, ({}, {})".format(mean_scaled_mae, lower_limit, upper_limit))


origin_all_mse = np.concatenate(origin_all_mse).reshape(-1)
mean_origin_mse = np.mean(origin_all_mse)
std_origin_mse = np.std(origin_all_mse)
lower_limit, upper_limit = mean_origin_mse - norm.ppf(0.95) * std_origin_mse/np.sqrt(origin_all_mse.shape[0]), mean_origin_mse + norm.ppf(0.95) * std_origin_mse/np.sqrt(origin_all_mse.shape[0])
print("Origin mse:{}, ({}, {})".format(mean_origin_mse, lower_limit, upper_limit))

origin_all_mae = np.concatenate(origin_all_mae).reshape(-1)
mean_origin_mae = np.mean(origin_all_mae)
std_origin_mae = np.std(origin_all_mae)
lower_limit, upper_limit = mean_origin_mae - norm.ppf(0.95) * std_origin_mae/np.sqrt(origin_all_mae.shape[0]), mean_origin_mae + norm.ppf(0.95) * std_origin_mae/np.sqrt(origin_all_mae.shape[0])
print("Origin mae:{}, ({}, {})".format(mean_origin_mae, lower_limit, upper_limit))







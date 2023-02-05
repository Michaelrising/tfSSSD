import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from utils.util import get_mask_holiday
from imputers.MegaModel import MegaImputer
from imputers.SSSDImputer import SSSDImputer
from imputers.CSDIImputer import CSDIImputer

class Imputer:
    def __init__(self,
                 model_path,
                 log_path,
                 model,
                 alg=None,
                 seq_len=200,
                 in_feature=5,
                 out_feature=5,
                 *args,
                 **kwargs):

        path_name = '/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "_seq_{}".format(seq_len)
        self.model_path = model_path + path_name
        self.log_path = log_path + path_name

        self.seq_len = seq_len

        if model == 'sssd':
            self.model = self.construct_sssd_s4(in_feature, out_feature, alg=alg)
        elif model =='mega':
            self.model = self.construct_mega_model(in_feature, out_feature)
        elif model == 'csdi':
            self.model = self.construct_csdi_model(in_feature, out_feature, alg=alg)
        else: raise ValueError

    def train(self,
              epoch=50,
              batch_size=32,
              lr=1e-3,
              validation_split=0.1,
              step_lr=False,
              amsgrad=False,
              save_weights_only=True):
        # get data
        training_data, training_mask = self.prepare_data(self.seq_len) # L B K
        X, Y = self.prepare_x_y(training_data, training_mask)
        L, N, K = training_data.shape  # C is the dimension of each audio, L is audio length, N is the audio batch

        # define optimizer
        if step_lr:
            p1 = int(0.6 * epoch * N / batch_size)
            p2 = int(0.75 * epoch * N / batch_size)
            # p3 = int(0.8 * self.epochs * series.shape[0] / self.batch_size)
            boundaries = [p1]
            values = [lr, lr * 0.1]

            learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        else:
            learning_rate_fn = lr
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-6, amsgrad=amsgrad)

        # define loss
        if isinstance(self.model, CSDIImputer):
            loss = None
        else:
            loss = keras.losses.MeanSquaredError()

        # define callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=1)
        earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
        best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=save_weights_only,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        )

        # training
        self.model.compile(optimizer=optimizer, loss=loss)
        history = self.model.fit(
            x=X,
            y=Y,
            batch_size=batch_size,
            epochs=epoch,
            validation_split=validation_split,
            callbacks=[tensorboard_callback,
                       earlyStop_loss_callback,
                       best_checkpoint_callback],
        )
        plt.plot(history.history["loss"], c='blue', label='Loss')
        plt.plot(history.history["val_loss"], c='orange', label='Val_loss')
        plt.grid()
        plt.legend()
        plt.title("Training Loss")
        plt.savefig(self.log_path + '/training.png')
        plt.show()
        self.model.summary()

    def prepare_x_y(self, training_data, training_mask):
        observed_mask = (~np.isnan(training_data)).astype(np.float32)
        training_data = np.nan_to_num(training_data)
        training_data = tf.transpose(tf.convert_to_tensor(training_data, dtype=tf.float32),
                                     perm=[1, 2, 0])  # batch dim # L N C -> [N C L]
        training_mask = tf.transpose(tf.convert_to_tensor(training_mask, dtype=tf.float32),
                                     perm=[1, 2, 0])  # batch dim # L N C -> [N C L]
        observed_mask = tf.transpose(tf.convert_to_tensor(observed_mask, dtype=tf.float32),
                                     perm=[1, 2, 0]) # batch dim # L N C -> [N C L]
        loss_mask = tf.cast(1. - training_mask, tf.bool)
        if isinstance(self.model, SSSDImputer):
            noise = tf.random.normal(shape=tf.shape(training_data), dtype=training_data.dtype)
            diffusion_steps = tf.random.uniform(shape=(tf.shape(training_data)[0], 1, 1), maxval=self.config['T'],
                                                dtype=tf.int32)  # randomly sample diffusion steps from 1~T
            X = [noise, training_data, training_mask, loss_mask, diffusion_steps]
            Y = None
        elif isinstance(self.model, MegaImputer):
            loss_mask = tf.cast(observed_mask - training_mask, tf.bool) # Mega is not generative model, so the real missing data should be masked in loss
            X = [training_data, training_mask, loss_mask]
            Y = training_data
        elif  isinstance(self.model, CSDIImputer):
            X = [training_data, training_mask, loss_mask]
            Y = None
        else:
            X, Y = None, None
        return X, Y

    def prepare_data(self, seq_len):
        training_data = []
        training_mask = []
        for ticker in ['DJ', 'ES', 'SE']:
            data_name = ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
            ticker_data = np.load('../datasets/Stocks/' + data_name, allow_pickle=True).astype(np.float32)  # L N C
            scalar0 = MinMaxScaler()
            ticker_data = np.array([scalar0.fit_transform(tk) for tk in ticker_data.transpose([1, 0, 2])]).transpose(
                [1, 0, 2])  # N L C -> L N C
            # generate masks: observed_masks + man_made mask
            ticker_mask = get_mask_holiday(ticker_data)  # N L C
            ticker_mask = ticker_mask.numpy().transpose([1, 0, 2])  # L N C
            for i in range(ticker_data.shape[0] // seq_len):
                ticker_chunk = ticker_data[seq_len * i:seq_len * (i + 1)]  # L N C
                training_data.append(ticker_chunk)
                training_mask.append(ticker_mask[seq_len * i:seq_len * (i + 1)])
            np.save(self.log_path + '/' + ticker + '_all_stocks_2013-01-02_to_2023-01-01_gt_masks.npy', ticker_mask)
        training_data = np.concatenate(training_data, axis=1).astype(float)  # L B K
        training_mask = np.concatenate(training_mask, axis=1).astype(np.float32)  # L B K
        training_all = np.concatenate((training_data.transpose([1, 0, 2]), training_mask.transpose([1, 0, 2])),
                                      axis=-1)  # B L 2*K
        # shuffle data
        np.random.shuffle(training_all)
        training_data = training_all[..., :5].transpose([1, 0, 2])  # L B K
        training_mask = training_all[..., 5:].transpose([1, 0, 2])  # L B K
        print('Loading stocks data: all!')
        print(training_data.shape)

        print('Data loaded')
        return training_data, training_mask

    def construct_sssd_s4(self,
                          in_feature,
                          out_feature,
                          T=200,
                          beta_0=0.001,
                          beta_1=0.02,
                          num_res_layers=36,
                          res_channels=256,
                          skip_channels=256,
                          s4_lmax=200,
                          alg='S4',
                          ):
        self.config = {}
        self.config["T"] = T
        self.config["beta_0"] = beta_0
        self.config["beta_T"] = beta_1
        self.config["in_channels"] = in_feature
        self.config["out_channels"] = out_feature
        self.config["num_res_layers"] = num_res_layers
        self.config["res_channels"] = res_channels
        self.config["skip_channels"] = skip_channels
        self.config["diffusion_step_embed_dim_in"] = 128
        self.config["diffusion_step_embed_dim_mid"] = 512
        self.config["diffusion_step_embed_dim_out"] = 512
        self.config["s4_lmax"] = s4_lmax
        self.config["s4_d_state"] = 64
        self.config["s4_dropout"] = 0.0
        self.config["s4_bidirectional"] = 1
        self.config["s4_layernorm"] = 1
        self.config["alg"] = alg
        self.config["only_generate_missing"] = 1

        self.log_path = self.log_path + "_T_{}_Layers_{}".format(T, num_res_layers)
        self.model_path += "_T_{}_Layers_{}/".format(T, num_res_layers)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        config_filename = self.log_path  + '/config_SSSD_stocks' + "_T_{}_Layers_{}".format(T, num_res_layers) +'.json'
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(self.config, f, indent=4)

        model = SSSDImputer(**self.config)
        return model

    def construct_mega_model(self,
                             in_feature,
                             out_feature,
                             mid_features=128,
                             depth=8,
                             chunk_size=-1,
                             pre_norm=True
                             ):
        self.config = {}
        self.config['in_feature'] = in_feature
        self.config['mid_features'] = mid_features  # original name is dim
        self.config['depth'] = depth
        self.config['out_features'] = out_feature
        self.config['chunk_size'] = chunk_size
        self.config['ff_mult'] = 2
        self.config['pre_norm'] = pre_norm

        self.log_path += '/'
        self.model_path += '/'
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        model = MegaImputer(**self.config)
        config_filename = self.log_path+'/config_mega.json'
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(self.config, f, indent=4)

        return model

    def construct_csdi_model(self,
                             in_feature,
                             out_feature,
                             masking='rm',
                             missing_ratio_or_k=0.1,
                             epochs=50,
                             batch_size=64,
                             lr=1.0e-3,
                             layers=4,
                             channels=64,
                             nheads=8,
                             difussion_embedding_dim=128,
                             beta_start=0.0001,
                             beta_end=0.5,
                             num_steps=200,
                             lmax=100,
                             schedule='quad',
                             alg='transformer',
                             is_unconditional=0,
                             timeemb=128,
                             featureemb=16,
                             ):
        self.config = {}
        self.config['train'] = {}
        self.config['train']['epochs'] = epochs
        self.config['train']['batch_size'] = batch_size
        self.config['train']['lr'] = lr
        self.config['train']['path_save'] = self.model_path

        self.config['diffusion'] = {}
        self.config['diffusion']['layers'] = layers
        self.config['diffusion']['channels'] = channels
        self.config['diffusion']['nheads'] = nheads
        self.config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        self.config['diffusion']['beta_start'] = beta_start
        self.config['diffusion']['beta_end'] = beta_end
        self.config['diffusion']['num_steps'] = num_steps
        self.config['diffusion']['schedule'] = schedule
        self.config['diffusion']['time_layer'] = alg
        self.config['diffusion']['lmax'] = lmax

        self.config['model'] = {}
        self.config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        self.config['model']['is_unconditional'] = is_unconditional
        self.config['model']['timeemb'] = timeemb
        self.config['model']['target_dim'] = out_feature
        self.config['model']['featureemb'] = featureemb
        self.config['model']['masking'] = masking

        print(json.dumps(self.config, indent=4))
        self.log_path += '/'
        self.model_path += '/'
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        config_filename = self.log_path + '/config_csdi_training_holiday.json'
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(self.config, f, indent=4)
        model = CSDIImputer(self.config)

        return model

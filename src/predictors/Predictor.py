import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from imputers.SSSDImputer import SSSDImputer
from imputers.MegaModel import MegaImputer
from sklearn.preprocessing import MinMaxScaler


class Predictor:
    def __init__(self,
                 model_path,
                 log_path,
                 in_channels=2,
                 out_channels=2,
                 model=None,
                 *args,
                 **kwargs):
        self.model_path = model_path
        self.log_path = log_path

        if model == 'sssd':
            self.model = self.construct_sssd_s4(in_channels=in_channels,
                                                out_channels=out_channels)
        elif model == 'mega':
            self.model = self.construct_mega_model(in_feature=in_channels,
                                                   out_features=out_channels)
        else: raise ValueError

    def train(self,
              data_path,
              lr=1e-3,
              amsgrad=True,
              batch_size=64,
              epochs=50,
              seq_len=200,
              infer_flag=False):

        # define optimizer
        optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-6, clipnorm=0.5, amsgrad=amsgrad)
        # define loss
        loss = keras.losses.MeanSquaredError()
        # define callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=1)
        earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
        # earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10)
        best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_format='tf'
        )

        training_data, training_mask = self.prepare_data(data_path, seq_len)
        print('Data loaded')
        # prepare X

        L, N, K = training_data.shape  # C is the dimension of each audio, L is audio length, N is the audio batch
        print("missing rate:" + str(1 - np.sum(training_mask) / (N * K * L)))

        X, Y = self.prepare_x_y(training_data, training_mask)

        if not infer_flag:
            self.model.compile(optimizer=optimizer, loss=loss)
            history = self.model.fit(x=X,
                                    y=Y,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_split=0.1,
                                    callbacks=[tensorboard_callback,
                                                earlyStop_loss_callback,
                                                best_checkpoint_callback
                                                ]
                                     )

            plt.plot(history.history["loss"], c='blue')
            plt.plot(history.history["val_loss"], c='orange')
            plt.grid()
            plt.title("Loss")
            plt.savefig(self.log_path + '/loss.png')
            plt.show()

        self.model.summary()

    def prepare_data(self, data_path, seq_len):
        # prepare data set
        training_data = []
        training_mask = []
        for ticker in ['DJ', 'ES', 'SE']:
            data_name = "/generated_scaled_" + ticker + '_all_stocks_2013-01-02_to_2023-01-01.npy'
            ticker_data = np.load(data_path + data_name, allow_pickle=True).astype(np.float32)  # L N C
            for i in range((ticker_data.shape[0]) // seq_len):
                ticker_chunk = ticker_data[seq_len * i:seq_len * (i + 1)]  # L N C
                mask_chunk = np.ones_like(ticker_chunk)
                if ticker == 'SE':
                    ticker_chunk = ticker_data[seq_len * i:seq_len * (i + 1)]  # L N C
                    mask_chunk = np.ones_like(ticker_chunk)
                    mask_chunk[-1] = np.zeros_like(ticker_chunk[0])
                training_data.append(ticker_chunk)
                training_mask.append(mask_chunk)
        training_data = np.concatenate(training_data, axis=1).astype(float)  # L B K
        training_mask = np.concatenate(training_mask, axis=1).astype(np.float32)  # L B K
        training_all = np.concatenate((training_data.transpose([1, 0, 2]), training_mask.transpose([1, 0, 2])),
                                      axis=-1)  # B L 2*K
        # shuffle data
        np.random.shuffle(training_all)
        training_data = training_all[..., :5].transpose([1, 0, 2])  # L B K
        training_mask = training_all[..., 5:].transpose([1, 0, 2])  # L B K
        print('Loading stocks data: ' + ticker)
        print(training_data.shape)
        return training_data, training_mask

    def prepare_x_y(self, training_data, training_mask):
        training_data = tf.transpose(tf.convert_to_tensor(training_data, dtype=tf.float32),
                                     perm=[1, 2, 0])  # batch dim # L N C -> [N C L]
        training_mask = tf.transpose(tf.convert_to_tensor(training_mask, dtype=tf.float32),
                                     perm=[1, 2, 0])  # batch dim # L N C -> [N C L]
        loss_mask = tf.cast(1.0 - training_mask, tf.bool)
        if isinstance(self.model, SSSDImputer):
            noise = tf.random.normal(shape=tf.shape(training_data), dtype=training_data.dtype)
            diffusion_steps = tf.random.uniform(shape=(tf.shape(training_data)[0], 1, 1), maxval=self.config['T'],
                                                dtype=tf.int32)  # randomly sample diffusion steps from 1~T
            X = [noise, training_data, training_mask, loss_mask, diffusion_steps]
            Y = None
        elif isinstance(self.model, MegaImputer):
            X = [training_data, training_mask, loss_mask]
            Y = training_data
        else:
            X, Y = None, None
        return X, Y

    def construct_sssd_s4(self,
                          in_channels,
                          out_channels,
                          T = 200,
                          beta_0=0.001,
                          beta_1=0.02,
                          num_res_layers=36,
                          res_channels=256,
                          skip_channels=256,
                          s4_lmax=200,
                          ):

        self.config = {}
        self.config["T"]= T
        self.config["beta_0"] = beta_0
        self.config["beta_T"]= beta_1
        self.config["in_channels"] = in_channels
        self.config["out_channels"] = out_channels
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
        self.config["alg"] = 'S4'
        self.config["only_generate_missing"] = 1

        model = SSSDImputer(**self.config)
        return model

    def construct_mega_model(self,
                            in_feature,
                            out_features,
                            mid_features=128,
                            depth=8,
                            chunk_size=-1,
                            pre_norm=True
                            ):

        self.config = {}
        self.config['in_feature'] = in_feature
        self.config['mid_features'] = mid_features  # original name is dim
        self.config['depth'] = depth
        self.config['out_features'] = out_features
        self.config['chunk_size'] = chunk_size
        self.config['ff_mult'] = 2
        self.config['pre_norm'] = pre_norm
        model = MegaImputer(**self.config)

        return model


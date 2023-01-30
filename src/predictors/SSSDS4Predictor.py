from src.imputers.SSSDImputer import Conv, Residual_group, ZeroConv1d
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class SSSD(keras.Model):
    def __init__(self,
                 in_channels,
                 res_channels,
                 skip_channels,
                 out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 alg):
        super(SSSD, self).__init__()

        self.init_conv = keras.Sequential()  # initial process for input
        self.init_conv.add(keras.layers.Input(shape=(in_channels, None,)))
        self.init_conv.add(Conv(in_channels, res_channels, kernel_size=1))
        self.init_conv.add(keras.layers.Activation(keras.activations.relu))

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm,
                                             alg=alg,
                                             )

        # self.final_conv = keras.Sequential()
        # self.final_conv.add(keras.layers.Input(shape=(skip_channels, None,)))
        # self.final_conv.add(Conv(in_channels=skip_channels, out_channels=skip_channels, kernel_size=1))
        # self.final_conv.add(keras.layers.Activation(keras.activations.relu))
        # self.final_conv.add(ZeroConv1d(skip_channels, out_channels))

        self.output_f = keras.Sequential([
            keras.layers.Conv1D(filters=64, kernel_size=2, data_format='channels_last'),
            keras.layers.MaxPooling1D(pool_size=2, strides=2),
            # # keras.layers.LayerNormalization(axis=-1),
            keras.layers.Conv1D(filters=16, kernel_size=2, data_format='channels_last'),
            keras.layers.MaxPooling1D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=keras.activations.relu),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(out_channels, activation=keras.activations.relu)
        ])

    @tf.function
    def call(self, input_data, training=True):
        noise, conditional, mask, diffusion_steps = input_data

        conditional = conditional * mask
        conditional = tf.concat([conditional, mask], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.output_f(x)

        return y


class SSSDPredictor:
    def __init__(self,
                 model_path,
                 log_path,
                 in_channels=6,
                 out_channels=2,
                 *args,
                 **kwargs):
        self.model_path = model_path
        self.log_path = log_path
        self.config ={}
        self.config["in_channels"]= in_channels
        self.config["out_channels"]= out_channels
        self.config["num_res_layers"]= 18
        self.config["res_channels"]= 256
        self.config["skip_channels"]= 256
        self.config["diffusion_step_embed_dim_in"]= 128
        self.config["diffusion_step_embed_dim_mid"]= 512
        self.config["diffusion_step_embed_dim_out"]= 512
        self.config["s4_lmax"]= 100
        self.config["s4_d_state"]=64
        self.config["s4_dropout"]= 0.0
        self.config["s4_bidirectional"]= 1
        self.config["s4_layernorm"]= 1

        self.model = SSSD(**self.config)

    def train(self,
              data,
              lr=1e-3,
              amsgrad=True,
              batch_size=64,
              epochs=50,
              infer_flag=False):

        # define optimizer
        optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-6, amsgrad=amsgrad)
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

        # prepare data set
        dj30, es50, hs70 = data # the imputed data with shape 2609 x L x 6
        # DJ30 = np.load('../../datasets/Stocks/DJ_all_stocks_2013-01-02_to_2023-01-01.npy')
        # ES50 = np.load('../../datasets/Stocks/ES_all_stocks_2013-01-02_to_2023-01-01.npy')
        # HS70 = np.load('../../datasets/Stocks/SE_all_stocks_2013-01-02_to_2023-01-01.npy')
        X = np.concatenate((dj30, es50), axis=1) # L B H
        mask = ~np.isnan(X)
        cond = X

        # X = rearrange(X, 'l b h -> l (bh)') # L H'
        Y = tf.convert_to_tensor(hs70) # time_length * H

        # Visualize the training progress of the model.
        # re = self.model.call(X[:64])
        if not infer_flag:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                                     callbacks=[tensorboard_callback,
                                                earlyStop_loss_callback,
                                                best_checkpoint_callback
                                                ])

            plt.plot(history.history["loss"], c='blue')
            plt.plot(history.history["val_loss"], c='orange')
            plt.grid()
            plt.title("Loss")
            plt.savefig(self.log_path + '/loss.png')
            plt.show()

        # self.model.built_after_run()
        self.model.summary()


from imputers.MegaModel import Mega
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange


class Predictor:
    def __init__(self,
                 model_path,
                 log_path,
                 features=256,
                 mid_feature=64,
                 out_feature=2,
                 depth=8,
                 chunk_size=-1,
                 *args,
                 **kwargs):
        self.model_path = model_path
        self.log_path = log_path
        self.model = Mega(features=features,
                          mid_feature=mid_feature,
                          out_dim=out_feature,
                          depth=depth,
                          chunk_size=chunk_size)

    def train(self,
              data,
              lr=1e-4,
              amsgrad=False,
              batch_size=2,
              epochs=50,
              infer_flag=False):

        # define optimizer
        optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-6)
        # define loss
        loss = keras.losses.MeanSquaredError()
        # define callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=1)
        earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
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
        X, Y = data # the imputed data with shape 2609 x L x 6
        # DJ30 = np.load('../../datasets/Stocks/DJ_all_stocks_2013-01-02_to_2023-01-01.npy')
        # ES50 = np.load('../../datasets/Stocks/ES_all_stocks_2013-01-02_to_2023-01-01.npy')
        # HS70 = np.load('../../datasets/Stocks/SE_all_stocks_2013-01-02_to_2023-01-01.npy')
        X = tf.convert_to_tensor(X) # B L N' C
        Y = tf.convert_to_tensor(Y) # B N" 1

        # Visualize the training progress of the model.
        # re = self.model.call(X[:64])
        if not infer_flag:
            self.model.compile(optimizer=optimizer, loss=loss)
            history = self.model.fit(x=X, y=Y,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     validation_split=0.2,
                                     callbacks=[tensorboard_callback,
                                                earlyStop_loss_callback,
                                                best_checkpoint_callback
                                                ])

        #     plt.plot(history.history["loss"], c='blue')
        #     plt.plot(history.history["val_loss"], c='orange')
        #     plt.grid()
        #     plt.title("Loss")
        #     plt.savefig(self.log_path + '/loss.png')
        #     plt.show()
        #
        # # self.model.built_after_run()
        # self.model.summary()

    def evaluate(self, x):
        X = tf.convert_to_tensor(x)  # B L N' C
        prediction = self.model.predict(X)
        return prediction











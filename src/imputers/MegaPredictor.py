from .MegaModel import Mega
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


class MegaPredictor:
    def __init__(self,
                 model_path,
                 log_path,
                 features=64,
                 depth=8,
                 chunk_size=-1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.log_path = log_path
        self.model = Mega(features=features,
                          depth=depth,
                          chunk_size=chunk_size)

    def train(self,
              data,
              lr=1e-3,
              amsgrad=True,
              batch_size=32,
              epochs=50,
              infer_flag=False):

        # define optimizer
        optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-6, amsgrad=amsgrad, clipnorm=0.5)
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
        X = tf.convert_to_tensor(np.concatenate((dj30, es50), axis=1)) # B L1+L2 6
        Y = tf.convert_to_tensor(hs70)

        # Visualize the training progress of the model.
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





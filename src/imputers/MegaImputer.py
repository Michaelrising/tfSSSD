from MegaModel import Mega
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class MegaImputer:
    def __init__(self, model_path, log_path, config_path):
        self.model_path=model_path
        self.log_path = log_path
        self.config_path = config_path
        self.model = None

    def train(self,
              data,
              lr=1e-3,
              amsgrad = False,
              batch_size=64,
              epochs=100,
              masking='rm',
              infer_flag=False):

        self.model = Mega(num_tokens=256,
                          dim=512,
                          depth=8)

        optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-6, amsgrad=amsgrad)

        # define callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=1)
        earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
        # earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10)
        best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_format='tf'
        )

        # prepare data set



        # Visualize the training progress of the model.
        if not infer_flag:
            self.model.compile(optimizer=optimizer)
            history = self.model.fit(x=train_data, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                                     # validation_data=(validation_data,),
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
        # else:
        #     self.model.compile(optimizer=optimizer)
        #     pre_run_data = series[:self.batch_size]
        #     pre_run_data = TrainDataset(pre_run_data, missing_ratio_or_k=0.1, masking='rm')
        #     pre_run_data = self.process_data(pre_run_data)
        #     self.model.fit(x=pre_run_data, batch_size=self.batch_size, epochs=1)
        #     print('==' * 10 + 'Pre Train' + '==' * 10)
        self.model.built_after_run()
        self.model.summary()





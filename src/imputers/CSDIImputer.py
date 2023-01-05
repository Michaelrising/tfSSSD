import numpy as np

from .CSDI import *
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import time
from einops import rearrange
import tensorflow_probability as tfp

class CSDIImputer:
    def __init__(self, model_path, log_path, config_path,
              masking='rm',
              missing_ratio_or_k=0.1,
              epochs=50,
              batch_size=32,
              lr=1.0e-3,
              layers=4,
              channels=64,
              nheads=8,
              difussion_embedding_dim=128,
              beta_start=0.0001,
              beta_end=0.5,
              num_steps=50,
              schedule='quad',
              algo='transformer',
              is_unconditional=0,
              timeemb=128,
              featureemb=16,
              target_strategy='random',
                 ):

        '''

        :param model_path: save path for the result of model
        :param log_path: save path for the log file including tensorboard writer
        :param config_path: save path for the config file
        :param masking: the masking pattern
        :param missing_ratio_or_k:
        :param train_split:
        :param valid_split:
        :param epochs:
        :param samples_generate:
        :param batch_size:
        :param lr:
        :param layers:
        :param channels:
        :param nheads:
        :param difussion_embedding_dim:
        :param beta_start:
        :param beta_end:
        :param num_steps:
        :param schedule:
        :param is_unconditional:
        :param timeemb:
        :param featureemb:
        :param target_strategy:
        '''
        np.random.seed(0)
        random.seed(0)
        self.model_path = model_path
        self.log_path = log_path
        self.config_path = config_path
        self.batch_size = 16
        self.model = None
        self.epochs = epochs
        self.lr = lr

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
        self.config['diffusion']['time_layer'] = algo

        self.config['model'] = {}
        self.config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        self.config['model']['is_unconditional'] = is_unconditional
        self.config['model']['timeemb'] = timeemb
        self.config['model']['featureemb'] = featureemb
        self.config['model']['target_strategy'] = target_strategy
        self.config['model']['masking'] = masking

        print(json.dumps(self.config, indent=4))

        config_filename = self.config_path + "/config_csdi_training_" + masking
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(self.config, f, indent=4)

        if algo == 'S4':
            print('='*50)
            print("="*22 + 'CSDI-S4' + "="*22)
            print('=' * 50)
        else:
            print('=' * 50)
            print("=" * 17 + 'CSDI-TransFormer' + "=" * 17)
            print('=' * 50)
        '''
        CSDI imputer
        3 main functions:
        a) training based on random missing, non-random missing, and blackout masking.
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              series,
              validation_series=None,
              masking='rm'
              ):

        '''
        CSDI training function.


        Requiered parameters
        -series: Assumes series of shape (Samples, Length, Channels).
        -masking: 'rm': random missing, 'nrm': non-random missing, 'bm': black-out missing.
        -missing_ratio_or_k: missing ratio 0 to 1 for 'rm' masking and k segments for 'nrm' and 'bm'.
        -path_save: full path where to save model weights, configuration file, and means and std devs for de-standardization in inference.

        Default parameters
        -train_split: 0 to 1 representing the percentage of train set from whole data.
        -valid_split: 0 to 1. Is an adition to train split where 1 - train_split - valid_split = test_split (implicit in method).
        -epochs: number of epochs to train.
        -samples_generate: number of samples to be generated.
        -batch_size: batch size in training.
        -lr: learning rate.
        -layers: difussion layers.
        -channels: number of difussion channels.
        -nheads: number of difussion 'heads'.
        -difussion_embedding_dim: difussion embedding dimmensions.
        -beta_start: start noise rate.
        -beta_end: end noise rate.
        -num_steps: number of steps.
        -schedule: scheduler.
        -is_unconditional: conditional or un-conditional imputation. Boolean.
        -timeemb: temporal embedding dimmensions.
        -featureemb: feature embedding dimmensions.
        -target_strategy: strategy of masking.
        -wandbiases_project: weight and biases project.
        -wandbiases_experiment: weight and biases experiment or run.
        -wandbiases_entity: weight and biases entity.
        '''

        if masking != self.config['model']['masking']:
            self.config['model']['masking'] = masking

            config_filename = self.config_path + "/config_csdi_training_" + masking
            print('configuration file name:', config_filename)
            with open(config_filename + ".json", "w") as f:
                json.dump(self.config, f, indent=4)

        self.model = tfCSDI(series.shape[2], self.config)

        # define optimizer
        p1 = int(0.3 * self.epochs * series.shape[0] / self.batch_size)
        p2 = int(0.5 * self.epochs * series.shape[0] / self.batch_size)
        p3 = int(0.8 * self.epochs * series.shape[0] / self.batch_size)
        boundaries = [p1, p2, p3]
        values = [self.lr, self.lr * 0.1, self.lr * 0.1 * 0.1, self.lr * 0.1 * 0.1 * 0.1]

        learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-6)
        # define callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=1, profile_batch=10)
        earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10)
        earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10)
        best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_format='tf'
        )
        # prepare data set
        train_data = TrainDataset(series, missing_ratio_or_k=0.1,
                                  masking='rm')  # observed_values_tensor, observed_masks_tensor, gt_mask_tensor
        train_data = self.process_data(train_data) # observed_data, observed_mask, gt_mask, cond_mask
        if validation_series is not None:
            validation_data = TrainDataset(validation_series, missing_ratio_or_k=0.1,
                                  masking='rm')
            validation_data = self.process_data(validation_data)
        else:
            validation_data = None
        self.model.compile(optimizer=optimizer)
        # self.model.train_step((train_data, ))
        history = self.model.fit(x=train_data, batch_size=self.batch_size, epochs=self.epochs, validation_data=(validation_data, ),
                                callbacks=[tensorboard_callback,
                                         earlyStop_loss_callback,
                                         best_checkpoint_callback
                                         ])
        # model.save_weights(self.model_path, save_format='tf')
        # Visualize the training progress of the model.
        plt.plot(history.history["loss"], c='blue')
        plt.plot(history.history["val_loss"], c='orange')
        plt.grid()
        plt.title("Loss")
        plt.savefig(self.log_path + '/loss.png')
        plt.show()
        return train_data, validation_data # ,history

    def process_data(self, train_data):
        # observed_masks is the original missing, while gt_masks is the
        # original missing pattern plus the generated masks
        observed_data, observed_mask, gt_mask = train_data

        observed_data = tf.transpose(observed_data, perm=[0, 2, 1])
        observed_mask = tf.transpose(observed_mask, perm=[0, 2, 1])
        gt_mask = tf.transpose(gt_mask, [0, 2, 1])

        # cut_length = tf.zeros(observed_data.shape[0], dtype=tf.int64)
        for_pattern_mask = observed_mask
        if self.config["model"]["target_strategy"] != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        return observed_data, observed_mask, gt_mask, cond_mask

    def get_randmask(self, observed_mask):
        rand_for_mask = np.random.uniform(size=observed_mask.shape) * observed_mask.numpy()
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(observed_mask.shape[0]):
            sample_ratio = np.random.rand()
            num_observed = tf.reduce_sum(observed_mask[i]).numpy() #.sum().item()
            num_masked = round(num_observed * sample_ratio)
            index = tf.math.top_k(rand_for_mask[i], k=num_masked)
            index = index.indices.numpy()
            rand_for_mask[i][index] = -1
        cond_mask = tf.reshape(tf.convert_to_tensor(rand_for_mask > 0), observed_mask.shape)
        cond_mask = tf.cast(cond_mask, dtype=tf.float32)
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask.numpy()
        if self.config["model"]["target_strategy"] == "mix":
            rand_mask = self.get_randmask(observed_mask)
            rand_mask = rand_mask.numpy()

        cond_mask = observed_mask.numpy() #tf.identity(observed_mask)
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.config["model"]["target_strategy"] == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        cond_mask = tf.convert_to_tensor(cond_mask)
        cond_mask = tf.cast(cond_mask, dtype=tf.float32)
        return cond_mask

    def load_weights(self, path_config_name= "/config_csdi_training.json"):

        self.path_load_model_dic = self.model_path
        self.path_config = self.config_path + path_config_name

        '''
        Load weights and configuration file for inference.

        path_load_model: load model weights
        path_config: load configuration file
        '''
    def imputer(self,
               sample=None,
               gt_mask=None,
               ob_masks=None,
               n_samples=50,
               ):

        '''
        Imputation function
        sample: sample(s) to be imputed (Samples, Length, Channel)
        mask: mask where values to be imputed. 0's to impute, 1's to remain.
        n_samples: number of samples to be generated
        return imputations with shape (Samples, N imputed samples, Length, Channel)
        '''
        self.n_samples = tf.constant(n_samples, dtype=tf.int32)

        # prepare data set
        test_data = tf.stack(tf.split(sample, int(sample.shape[0]/self.batch_size), 0))
        test_gt_masks = tf.stack(tf.split(gt_mask, int(sample.shape[0]/self.batch_size), 0))
        test_ob_masks = tf.stack(tf.split(ob_masks,  int(sample.shape[0]/self.batch_size), 0))
        B, K, L = test_data[0].shape
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        # all_target = tf.TensorArray(dtype=tf.float32, size=int(sample.shape[0]/self.batch_size))
        # all_observed_point = tf.TensorArray(dtype=tf.float32, size=int(sample.shape[0]/self.batch_size))
        # all_observed_time = tf.TensorArray(dtype=tf.int32, size=int(sample.shape[0]/self.batch_size))
        # all_evalpoint = tf.TensorArray(dtype=tf.float32, size=int(sample.shape[0]/self.batch_size))
        # all_generated_samples = tf.TensorArray(dtype=tf.float32, size=int(sample.shape[0]/self.batch_size))

        @tf.function
        def single_batch_imputer(test_batch):
            # test_batch = (test_d, test_ob_m, test_gt_m)
            # observed_data, observed_mask, gt_mask
            samples, c_target, eval_points, observed_points, observed_time = self.model.impute(test_batch,
                                                                                               self.n_samples)
            # samples: n_samples B K L
            # samples = rearrange(samples, 'i j k l -> j i l k')  # (B,nsample,L,K)
            c_target = rearrange(c_target, 'i j k -> i k j')  # (B,L,K)
            eval_points = rearrange(eval_points, 'i j k -> i k j')

            samples_median = rearrange(tfp.stats.percentile(samples, 50., axis=0), 'i j k -> i k j')  # B K L -> B L K

            mse_current = (((samples_median - c_target) * eval_points) ** 2)
            mae_current = (tf.abs((samples_median - c_target) * eval_points))

            return samples #, tf.reduce_sum(mse_current), tf.reduce_sum(mae_current), tf.reduce_sum(eval_points)

        all_generated_samples = tf.stop_gradient(
                tf.map_fn(fn = single_batch_imputer, elems=(test_data, test_ob_masks, test_gt_masks),
                            fn_output_signature=tf.TensorSpec(shape=[n_samples,B, K, L], dtype=tf.float32),
                                                # tf.TensorSpec(shape=(), dtype=tf.float32),
                                                # tf.TensorSpec(shape=(), dtype=tf.float32),
                                                # tf.TensorSpec(shape=(), dtype=tf.float32)),
                            parallel_iterations=50,
                      )
        )

        # all_start_time = time.time()
        # for i in range(int(sample.shape[0]/self.batch_size)):
        #     test_batch = (test_data[i], test_ob_masks[i], test_gt_masks[i])
        #     # observed_data, observed_mask, gt_mask
        #     ite_start_time = time.time()
        #     samples, c_target, eval_points, observed_points, observed_time = self.model.impute(test_batch, self.n_samples)
        #     # samples: n_samples B K L
        #     ite_end_time = time.time()
        #     print('Ite-{} uses {} s'.format(i, int(ite_end_time - ite_start_time)))
        #     # samples = rearrange(samples, 'i j k l -> j i l k')  # (B,nsample,L,K)
        #     c_target = rearrange(c_target, 'i j k -> i k j')  # (B,L,K)
        #     # c_target = c_target.permute(0, 2, 1)
        #     eval_points = rearrange(eval_points, 'i j k -> i k j')
        #     # eval_points = eval_points.permute(0, 2, 1)
        #     # observed_points = rearrange(observed_points, 'i j k -> i k j')
        #     # observed_points = observed_points.permute(0, 2, 1)
        #
        #     samples_median = rearrange(tfp.stats.percentile(samples, 50., axis=0), 'i j k -> i k j') # B K L -> B L K
        #     # all_target.write(i, c_target)
        #     # all_evalpoint.write(i, eval_points)
        #     # all_observed_point.write(i, observed_points)
        #     # all_observed_time.write(i, observed_time)
        #     all_generated_samples = all_generated_samples.write(i, samples)
        #
        #     mse_current = (((samples_median - c_target) * eval_points) ** 2)
        #     mae_current = (tf.abs((samples_median - c_target) * eval_points))
        #
        #     mse_total += tf.reduce_sum(mse_current)
        #     mae_total += tf.reduce_sum(mae_current)
        #     evalpoints_total += tf.reduce_sum(eval_points)
        # all_end_time = time.time()
        # print("Total imputation uses time {} s".format(int(all_end_time - all_start_time)))

        return all_generated_samples #, mse_total, mae_total, evalpoints_total


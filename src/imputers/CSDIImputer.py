import numpy as np
import tensorflow_addons as tfa
from .CSDI import *
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import time
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

class CSDIImputer:
    def __init__(self, model_path, log_path, config_path,
              masking='rm',
              missing_ratio_or_k=0.1,
              epochs=50,
              batch_size=64,
              lr=5.0e-4,
              layers=4,
              channels=64,
              nheads=8,
              difussion_embedding_dim=128,
              beta_start=0.0001,
              beta_end=0.5,
              num_steps=200,
              lmax=100,
              schedule='quad',
              algo='transformer',
              is_unconditional=0,
              timeemb=128,
              featureemb=16,
              target_strategy='random',
              amsgrad=False,
              training=True,
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
        self.batch_size = batch_size
        self.model = None
        self.epochs = epochs
        self.lr = lr
        self.missing_ratio_or_k = missing_ratio_or_k
        self.amsgrad=amsgrad

        config_filename = self.config_path + "config_csdi_training_" + masking
        if training:
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
            self.config['diffusion']['lmax'] = lmax

            self.config['model'] = {}
            self.config['model']['missing_ratio_or_k'] = missing_ratio_or_k
            self.config['model']['is_unconditional'] = is_unconditional
            self.config['model']['timeemb'] = timeemb
            self.config['model']['featureemb'] = featureemb
            self.config['model']['target_strategy'] = target_strategy
            self.config['model']['masking'] = masking

            print(json.dumps(self.config, indent=4))

            print('configuration file name:', config_filename)
            with open(config_filename + ".json", "w") as f:
                json.dump(self.config, f, indent=4)
        else:

            with open(config_filename + '.json') as f:
                data = f.read()
            self.config = json.loads(data)

        # parameters for diffusion models
        self.num_steps = num_steps
        if schedule == "quad":
            self.beta = tf.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_steps) ** 2
        elif schedule == "linear":
            self.beta = tf.linspace(beta_start, beta_end, self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = tf.math.cumprod(self.alpha_hat)
        self.alpha_tf = tf.expand_dims(tf.expand_dims(tf.cast(self.alpha, dtype=tf.float32), 1), 1)

        if algo == 'S4':
            print('='*50)
            print("="*22 + 'CSDI-S4' + "="*21)
            print('=' * 50)
        elif algo=='transformer':
            print('=' * 50)
            print("=" * 17 + 'CSDI-TransFormer' + "=" * 17)
            print('=' * 50)
        elif algo == 'S5':
            print('='*50)
            print("="*22 + 'CSDI-S5' + "="*21)
            print('=' * 50)
        elif algo == 'Mega':
            print('=' * 50)
            print("=" * 21 + 'CSDI-Mega' + "=" * 20)
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
              masking='rm',
              infer_flag=False,
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


        self.config['model']['masking'] = masking
        self.config['diffusion']['lmax'] = series.shape[1]
        config_filename = self.config_path + "/config_csdi_training_" + masking
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(self.config, f, indent=4)

        self.model = tfCSDI(series.shape[2], self.config)

        # define optimizer
        p1 = int(0.5 * self.epochs * series.shape[0] / self.batch_size)
        p2 = int(0.75 * self.epochs * series.shape[0] / self.batch_size)
        # p3 = int(0.8 * self.epochs * series.shape[0] / self.batch_size)
        boundaries = [p1, p2]
        values = [self.lr, self.lr * 0.1, self.lr * 0.1 * 0.1]

        learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-6, amsgrad=self.amsgrad)

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
        train_data = TrainDataset(series, missing_ratio_or_k=self.missing_ratio_or_k,
                                  masking=masking, batch_size=self.batch_size)  # observed_values_tensor, observed_masks_tensor, gt_mask_tensor
        train_data = self.process_data(train_data) # observed_data, observed_mask, gt_mask, cond_mask
        if validation_series is not None:
            validation_data = TrainDataset(validation_series, missing_ratio_or_k=self.missing_ratio_or_k,
                                  masking=masking)
            validation_data = self.process_data(validation_data)
        else:
            validation_data = None
        # self.model.compile(optimizer=optimizer)
        # pre_run_data = (train_data[0][:self.batch_size], train_data[1][:self.batch_size], train_data[2][:self.batch_size], train_data[3][:self.batch_size])
        # print('=='*10 + 'Pre Train' + '=='*10)
        # logdir = self.log_path + "/trace_log"
        # writer = tf.summary.create_file_writer(logdir)
        # tf.summary.trace_on(graph=True, profiler=True)
        # # Forward pass
        # self.model.train_step((pre_run_data, ))
        # self.model.test_step((pre_run_data, ))
        # with writer.as_default():
        #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
        # print('==' * 10 + 'Pre Train' + '==' * 10)
        # self.model.built_after_run()
        # self.model.summary()


        # Visualize the training progress of the model.
        if not infer_flag:
            self.model.compile(optimizer=optimizer)
            history = self.model.fit(x=train_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1,
                                     # validation_data=(validation_data,),
                                     callbacks=[tensorboard_callback,
                                                earlyStop_loss_callback,
                                                best_checkpoint_callback
                                                ])

            plt.plot(history.history["loss"], c='blue', label='Loss')
            plt.plot(history.history["val_loss"], c='orange', label='Val_loss')
            plt.grid()
            plt.legend()
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
        return train_data


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
            else: # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        cond_mask = tf.convert_to_tensor(cond_mask)
        cond_mask = tf.cast(cond_mask, dtype=tf.float32)
        return cond_mask

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
        sample = sample[:self.batch_size*int(sample.shape[0]//self.batch_size)]
        gt_mask = gt_mask[:self.batch_size*int(sample.shape[0]//self.batch_size)]
        ob_masks = ob_masks[:self.batch_size*int(sample.shape[0]//self.batch_size)]

        test_data = tf.stack(tf.split(sample, int(sample.shape[0]/self.batch_size), 0))
        test_gt_masks = tf.stack(tf.split(gt_mask, int(sample.shape[0]/self.batch_size), 0))
        test_ob_masks = tf.stack(tf.split(ob_masks,  int(sample.shape[0]/self.batch_size), 0))
        B, K, L = test_data[0].shape
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0
        all_generated_samples = []
        i = 0
        pbar = tqdm(total=sample.shape[0]/self.batch_size)
        while i < int(sample.shape[0]/self.batch_size):
            test_batch = (test_data[i], test_ob_masks[i], test_gt_masks[i])
            # observed_data, observed_mask, gt_mask
            samples, _, _, _, _ = self.impute(test_batch, self.n_samples)
            # samples: n_samples B K L
            all_generated_samples.append(samples)
            i += 1
            if i % 5 == 0 and i > 0:
                pbar.update(5)

        return tf.stack(all_generated_samples) #, mse_total, mae_total, evalpoints_total

    @tf.function
    def impute(self, batch, n_samples):
        observed_data, observed_mask, gt_mask = batch
        cond_mask = gt_mask
        B, K, L = observed_data.shape
        observed_tp = tf.reshape(tf.range(L), [1, L])  # 1 L
        observed_tp = tf.tile(observed_tp, [tf.shape(observed_data)[0], 1])  # B L
        target_mask = observed_mask - cond_mask
        side_info = tf.stop_gradient(
            self.model.get_side_info(observed_tp, cond_mask)
        )

        imputed_samples = tf.TensorArray(dtype=tf.float32, size=n_samples)
        sample_i = 0
        while sample_i < n_samples:
            current_sample = tf.TensorArray(dtype=tf.float32, size=1, clear_after_read=False)
            t = self.num_steps - 1
            current_sample = current_sample.write(0, tf.random.normal(observed_data.shape, dtype=observed_data.dtype))
            while t >= 0:
                # if self.is_unconditional == True:
                #     diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                #     diff_input = tf.expand_dims(diff_input, 1)  # (B,1,K,L)
                # else:
                cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
                noisy_target = tf.expand_dims((1 - cond_mask) * current_sample.read(0), 1)
                diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
                predicted = tf.stop_gradient(
                    self.model.diffmodel((diff_input, side_info, tf.constant([t])))
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = current_sample.write(0, coeff1 * (current_sample.read(0) - coeff2 * predicted))

                if t > 0:
                    noise = tf.random.normal(observed_data.shape, dtype=observed_data.dtype)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample = current_sample.write(0, current_sample.read(0) + sigma * noise)  # .mark_used()
                t -= 1
            imputed_samples = imputed_samples.write(sample_i, current_sample.read(0))
            sample_i += 1
        imputed_samples = imputed_samples.stack()
        return imputed_samples, observed_data, target_mask, observed_mask, observed_tp




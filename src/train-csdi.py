import numpy as np

from imputers.CSDI import *
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from einops import rearrange
import tensorflow_probability as tfp

class CSDIImputer:
    def __init__(self, device, model_path, log_path, config_path):
        np.random.seed(0)
        random.seed(0)
        self.device = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.log_path = log_path
        self.config_path = config_path
        self.batch_size = 16

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
              missing_ratio_or_k=0.1,
              train_split=0.7,
              valid_split=0.2,
              epochs=2,
              samples_generate=10,
              batch_size=16,
              lr=1.0e-3,
              layers=4,
              channels=64,
              nheads=8,
              difussion_embedding_dim=128,
              beta_start=0.0001,
              beta_end=0.5,
              num_steps=50,
              schedule='quad',
              is_unconditional=0,
              timeemb=128,
              featureemb=16,
              target_strategy='random',
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

        self.config = {}

        self.config['train'] = {}
        self.config['train']['epochs'] = epochs
        self.config['train']['batch_size'] = batch_size
        self.config['train']['lr'] = lr
        self.config['train']['train_split'] = train_split
        self.config['train']['valid_split'] = valid_split
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

        self.config['model'] = {}
        self.config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        self.config['model']['is_unconditional'] = is_unconditional
        self.config['model']['timeemb'] = timeemb
        self.config['model']['featureemb'] = featureemb
        self.config['model']['target_strategy'] = target_strategy
        self.config['model']['masking'] = masking

        print(json.dumps(self.config, indent=4))

        config_filename = self.config_path + "/config_csdi_training"
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(self.config, f, indent=4)

        model = tfCSDI(series.shape[2], self.config, self.device)

        # define optimizer
        p1 = int(0.5 * epochs * series.shape[0] / self.batch_size)
        p2 = int(0.65 * epochs * series.shape[0] / self.batch_size)
        p3 = int(0.85 * epochs * series.shape[0] / self.batch_size)
        boundaries = [p1, p2, p3]
        values = [lr, lr * 0.1, lr * 0.1 * 0.1, lr * 0.1 * 0.1 * 0.1]
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
        model.compile(optimizer=optimizer)
        history = model.fit(x=train_data, y=None, batch_size=self.batch_size, epochs=epochs, validation_data=None, #(validation_data, ),
                                callbacks=[tensorboard_callback,
                                         earlyStop_loss_callback,
                                         best_checkpoint_callback
                                         ])
        # model.save_weights(self.model_path, save_format='tf')
        # Visualize the training progress of the model.
        plt.plot(history.history["loss"])
        plt.grid()
        plt.title("Loss")
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
               device,
               sample=None,
               gt_mask=None,
               ob_masks=None,
               n_samples=100,
               ):

        '''
        Imputation function
        sample: sample(s) to be imputed (Samples, Length, Channel)
        mask: mask where values to be imputed. 0's to impute, 1's to remain.
        n_samples: number of samples to be generated
        return imputations with shape (Samples, N imputed samples, Length, Channel)
        '''
        self.device = device

        with open(self.config_path + "/config_csdi_training.json", "r") as f:
            config = json.load(f)

        # prepare data set
        test_data = tf.split(sample, int(sample.shape[0]/self.batch_size), 0)
        test_gt_masks = tf.split(gt_mask, int(sample.shape[0]/self.batch_size), 0)
        test_ob_masks = tf.split(ob_masks,  int(sample.shape[0]/self.batch_size), 0)
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []


        model = tfCSDI(sample.shape[1], self.config, self.device)

        # model.load_state_dict(torch.load((self.path_load_model_dic)))
        model.load_weights(self.model_path)

        for i in range(int(sample.shape[0]/self.batch_size)):
            test_batch = (test_data[i], test_ob_masks[i], test_gt_masks[i])
            # observed_data, observed_mask, gt_mask
            samples, c_target, eval_points, observed_points, observed_time = model.impute(test_batch, 10)
            # samples: n_samples B K L
            samples = rearrange(samples, 'i j k l -> j i l k')  # (B,nsample,L,K)
            c_target = rearrange(c_target, 'i j k -> i k j')  # (B,L,K)
            # c_target = c_target.permute(0, 2, 1)
            eval_points = rearrange(eval_points, 'i j k -> i k j')
            # eval_points = eval_points.permute(0, 2, 1)
            observed_points = rearrange(observed_points, 'i j k -> i k j')
            # observed_points = observed_points.permute(0, 2, 1)

            samples_median = tfp.stats.percentile(samples, 50., axis=1)
            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(observed_points)
            all_observed_time.append(observed_time)
            all_generated_samples.append(samples)

            mse_current = (((samples_median - c_target) * eval_points) ** 2)
            mae_current = (tf.abs((samples_median - c_target) * eval_points))

            mse_total += tf.reduce_sum(mse_current)
            mae_total += tf.reduce_sum(mae_current)
            evalpoints_total += tf.reduce_sum(eval_points)

        imputations = tf.concat(all_generated_samples)
        indx_imputation = tf.cast(~gt_mask, tf.bool)

        original_sample_replaced = []

        for original_sample, single_n_samples in zip(sample.numpy(),
                                                     imputations.numpy()):  # [x,x,x] -> [x,x] & [x,x,x,x] -> [x,x,x]
            single_sample_replaced = []
            for sample_generated in single_n_samples:  # [x,x] & [x,x,x] -> [x,x]
                sample_out = original_sample.copy()
                sample_out[indx_imputation] = sample_generated[indx_imputation]
                single_sample_replaced.append(sample_out)
            original_sample_replaced.append(single_sample_replaced)

        output = np.array(original_sample_replaced)

        return output


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    device = '/gpu:0'
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = '../results/mujoco/CSDI/' + current_time + '/csdi_model'
    log_path = '../log/mujoco/CSDI/' + current_time + '/csdi_log'
    config_path = './config'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    all_data = np.load('../datasets/Mujoco/train_mujoco.npy')
    # training_data = np.split(training_data, 160, 0)
    all_data = np.array(all_data)
    training_data = tf.convert_to_tensor(all_data[:1600])
    validation_data = tf.convert_to_tensor(all_data[1600:1920])
    print('Data loaded')
    CSDIImputer = CSDIImputer(device, model_path, log_path, config_path)
    train_data, validation_data = CSDIImputer.train(training_data, validation_data)
    # test_data = tf.convert_to_tensor(training_data[7000:])
    observed_data, ob_mask, gt_mask, _ = train_data
    CSDIImputer.imputer(device=device, sample=observed_data, gt_mask=gt_mask, ob_masks=ob_mask)

from imputers.CSDI import *
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class CSDIImputer:
    def __init__(self, device, model_path, log_path, config_path):
        np.random.seed(0)
        random.seed(0)
        self.device = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.log_path = log_path
        self.config_path = config_path

        '''
        CSDI imputer
        3 main functions:
        a) training based on random missing, non-random missing, and blackout masking.
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              series,
              masking='rm',
              missing_ratio_or_k=0.1,
              train_split=0.7,
              valid_split=0.2,
              epochs=200,
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
        # TODO keras compile fit and evaluate
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # define optimizer
        p1 = int(0.75 * epochs)
        p2 = int(0.9 * epochs)
        boundaries = [p1, p2]
        values = [lr, lr * 0.1, lr * 0.1 * 0.1]
        learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-6)
        # define callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=1)
        earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)
        earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=3)
        best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path + '/' + current_time,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True,
        )
        # prepare data set
        train_data = TrainDataset(series, missing_ratio_or_k=0.1,
                                  masking='rm')  # observed_values_tensor, observed_masks_tensor, gt_mask_tensor, timepoints
        train_data = self.process_data(train_data)
        model.compile(optimizer=optimizer)
        history = model.fit(x=train_data, batch_size=32, epochs=20, validation_split=0.,
                                callbacks=[tensorboard_callback,
                                         earlyStop_loss_callback,
                                         earlyStop_accu_call_back,
                                         best_checkpoint_callback])

        # Visualize the training progress of the model.
        plt.plot(history.history["loss"])
        plt.grid()
        plt.title("Loss")
        plt.show()

    def process_data(self, train_data):
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
            for_pattern_mask = observed_mask
        if self.config["model"]["target_strategy"] == "mix":
            rand_mask = self.get_randmask(observed_mask)
        # TODO tensor assignment
        cond_mask = tf.identity(observed_mask)
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.config["model"]["target_strategy"] == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def load_weights(self,
                     path_load_model='',
                     path_config=''):

        self.path_load_model_dic = path_load_model
        self.path_config = path_config

        '''
        Load weights and configuration file for inference.

        path_load_model: load model weights
        path_config: load configuration file
        '''
    #
    # def impute(self,
    #            sample,
    #            mask,
    #            device,
    #            n_samples=50,
    #
    #            ):
    #
    #     '''
    #     Imputation function
    #     sample: sample(s) to be imputed (Samples, Length, Channel)
    #     mask: mask where values to be imputed. 0's to impute, 1's to remain.
    #     n_samples: number of samples to be generated
    #     return imputations with shape (Samples, N imputed samples, Length, Channel)
    #     '''
    #
    #     if len(sample.shape) == 2:
    #         self.series_impute = tf.convert_to_tensor(np.expand_dims(sample, axis=0))
    #     elif len(sample.shape) == 3:
    #         self.series_impute = sample
    #
    #     self.device = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     with open(self.path_config, "r") as f:
    #         config = json.load(f)
    #
    #     testData = ImputeDataset(sample, mask)
    #     # test_loader = get_dataloader_impute(series=self.series_impute, len_dataset=len(self.series_impute),
    #     #                                     mask=mask, batch_size=config['train']['batch_size'])
    #
    #     # model = CSDI_Custom(config, self.device, target_dim=self.series_impute.shape[2])  # .to(self.device)
    #     model = tfCSDI(sample.shape[2], config, self.device)
    #
    #     # model.load_state_dict(torch.load((self.path_load_model_dic)))
    #     model.load(self.path_load_model_dic)
    #
    #     imputations = evaluate(model=model,
    #                            test_loader=test_loader,
    #                            nsample=n_samples,
    #                            scaler=1,
    #                            path_save='')
    #
    #     indx_imputation = tf.cast(~mask, tf.bool)
    #
    #     original_sample_replaced = []
    #
    #     for original_sample, single_n_samples in zip(sample.numpy(),
    #                                                  imputations):  # [x,x,x] -> [x,x] & [x,x,x,x] -> [x,x,x]
    #         single_sample_replaced = []
    #         for sample_generated in single_n_samples:  # [x,x] & [x,x,x] -> [x,x]
    #             sample_out = original_sample.copy()
    #             sample_out[indx_imputation] = sample_generated[indx_imputation]
    #             single_sample_replaced.append(sample_out)
    #         original_sample_replaced.append(single_sample_replaced)
    #
    #     output = np.array(original_sample_replaced)
    #
    #     return output


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    device = '/cpu:0'
    model_path = '../results/mujoco/CSDI'
    log_path = '../log/mujoco/CSDI'
    config_path = './config'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    training_data = np.load('../datasets/Mujoco/train_mujoco.npy')
    # training_data = np.split(training_data, 160, 0)
    training_data = np.array(training_data)
    training_data = tf.convert_to_tensor(training_data[:1000])
    print('Data loaded')
    CSDIImputer = CSDIImputer(device, model_path, log_path, config_path)
    CSDIImputer.train(training_data)

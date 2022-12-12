from imputers.CSDI import *
import matplotlib.pyplot as plt

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

        config = {}

        config['train'] = {}
        config['train']['epochs'] = epochs
        config['train']['batch_size'] = batch_size
        config['train']['lr'] = lr
        config['train']['train_split'] = train_split
        config['train']['valid_split'] = valid_split
        config['train']['path_save'] = self.model_path

        config['diffusion'] = {}
        config['diffusion']['layers'] = layers
        config['diffusion']['channels'] = channels
        config['diffusion']['nheads'] = nheads
        config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        config['diffusion']['beta_start'] = beta_start
        config['diffusion']['beta_end'] = beta_end
        config['diffusion']['num_steps'] = num_steps
        config['diffusion']['schedule'] = schedule

        config['model'] = {}
        config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        config['model']['masking'] = masking

        print(json.dumps(config, indent=4))

        config_filename = self.config_path + "/config_csdi_training"
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(config, f, indent=4)

        model = tfCSDI(series.shape[2], config, self.device)
        # TODO keras compile fit and evaluate
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
        earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)
        best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path + '/' + current_time,
            save_weights_only=False,
            monitor='accuracy',
            mode='max',
            save_best_only=True,
        )
        # prepare data set
        train_data = TrainDataset(series, missing_ratio_or_k=0.1,
                                  masking='rm')  # observed_values_tensor, observed_masks_tensor, gt_mask_tensor, timepoints

        model.compile(optimizer=optimizer)
        history = model.fit(x=train_data, batch_size=64, epochs=100, validation_split=0.1,
                                callbacks=[tensorboard_callback,
                                         earlyStop_loss_callback,
                                         earlyStop_accu_call_back,
                                         best_checkpoint_callback])

        # Visualize the training progress of the model.
        plt.plot(history.history["loss"])
        plt.grid()
        plt.title("Loss")
        plt.show()

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
    device = '/gpu:1'
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
    training_data = tf.convert_to_tensor(training_data)
    print('Data loaded')
    CSDIImputer = CSDIImputer( device, model_path, log_path, config_path)
    CSDIImputer.train(training_data)

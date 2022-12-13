import keras

from .CSDI_base import *

''' Standalone CSDI imputer. The imputer class is located in the last part of the notebook, please see more documentation there'''

# TODO
# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, path_save=""):
#     with torch.no_grad():
#         model.eval()
#         mse_total = 0
#         mae_total = 0
#         evalpoints_total = 0
#
#         all_target = []
#         all_observed_point = []
#         all_observed_time = []
#         all_evalpoint = []
#         all_generated_samples = []
#         with tqdm(test_loader, mininterval=5.0, maxinterval=5.0) as it:
#             for batch_no, test_batch in enumerate(it, start=1):
#                 output = model.evaluate(test_batch, nsample)
#
#                 samples, c_target, eval_points, observed_points, observed_time = output
#                 samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
#                 c_target = c_target.permute(0, 2, 1)  # (B,L,K)
#                 eval_points = eval_points.permute(0, 2, 1)
#                 observed_points = observed_points.permute(0, 2, 1)
#
#                 samples_median = samples.median(dim=1)
#                 all_target.append(c_target)
#                 all_evalpoint.append(eval_points)
#                 all_observed_point.append(observed_points)
#                 all_observed_time.append(observed_time)
#                 all_generated_samples.append(samples)
#
#                 mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)
#                 mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler
#
#                 mse_total += mse_current.sum().item()
#                 mae_total += mae_current.sum().item()
#                 evalpoints_total += eval_points.sum().item()
#
#                 it.set_postfix(ordered_dict={
#                         "rmse_total": np.sqrt(mse_total / evalpoints_total),
#                         "mae_total": mae_total / evalpoints_total,
#                         "batch_no": batch_no}, refresh=True)
#
#             with open(f"{path_save}generated_outputs_nsample"+str(nsample)+".pk","wb") as f:
#                 all_target = torch.cat(all_target, dim=0)
#                 all_evalpoint = torch.cat(all_evalpoint, dim=0)
#                 all_observed_point = torch.cat(all_observed_point, dim=0)
#                 all_observed_time = torch.cat(all_observed_time, dim=0)
#                 all_generated_samples = torch.cat(all_generated_samples, dim=0)
#
#                 pickle.dump(
#                     [
#                         all_generated_samples,
#                         all_target,
#                         all_evalpoint,
#                         all_observed_point,
#                         all_observed_time,
#                         scaler,
#                         mean_scaler,
#                     ],
#                     f,
#                 )
#
#             CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
#
#             with open(f"{path_save}result_nsample" + str(nsample) + ".pk", "wb") as f:
#                 pickle.dump(
#                     [
#                         np.sqrt(mse_total / evalpoints_total),
#                         mae_total / evalpoints_total,
#                         CRPS
#                     ],
#                     f)
#                 print("RMSE:", np.sqrt(mse_total / evalpoints_total))
#                 print("MAE:", mae_total / evalpoints_total)
#                 print("CRPS:", CRPS)
#
#
#     return all_generated_samples.cpu().numpy()
#

class tfCSDI(keras.Model):
    def __init__(self, target_dim, config, device):
        super(tfCSDI, self).__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        self.embed_layer = keras.Sequential()
        self.embed_layer.add(keras.layers.Input(shape=(None, self.target_dim, )))
        self.embed_layer.add(keras.layers.Embedding(input_dim=self.target_dim, output_dim=self.emb_feature_dim))

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = tf.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5,
                                    self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = tf.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = tf.math.cumprod(self.alpha_hat)  # TODO numpy default is flattened need to check shape
        self.alpha_tf = tf.expand_dims(tf.expand_dims(tf.cast(self.alpha, dtype=tf.float32), 1), 1)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def time_embedding(self, pos, d_model=128):
        pe = tf.zeros(pos.shape[0], pos.shape[1], d_model)  # .to(self.device)
        position = tf.expand_dims(pos, 2)
        div_term = 1 / tf.pow(10000.0, tf.range(0, d_model, 2) / d_model)  # .to(self.device)
        pe[:, :, 0::2] = tf.math.sin(position * div_term)
        pe[:, :, 1::2] = tf.math.cos(position * div_term)
        return pe


    def get_side_info(self, time_embed, cond_mask, time_fea):

        B, K, L = cond_mask.shape

        time_embed = tf.expand_dims(tf.transpose(time_embed, [0, 2, 1]), 2)  # .expand(-1, -1, K, -1)
        time_embed = tf.tile(time_embed, [1, 1, K, 1])
        feature_embed = self.embed_layer(tf.transpose(time_fea, [0, 2, 1]) )# .to(self.device))  # (K,emb)
        side_info = tf.concat([time_embed, feature_embed], axis=-1)  # (B,L,K,*)
        side_info = tf.transpose(side_info, perm=[0, 3, 2, 1])  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask, 1)  # (B,1,K,L)
            side_info = tf.concat([side_info, side_mask], axis=1)

        return side_info

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            loss_sum += loss  # .detach()

        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = tf.cast(tf.ones(B) * set_t, dtype=tf.int64)  # .long().to(self.device)
        else:
            # t = torch.randint(0, self.num_steps, [B]).to(self.device)
            t = tf.random.uniform(shape=(B,), minval=0, maxval=self.num_steps + 1, dtype=tf.int32)
        current_alpha = self.alpha_tf[t]  # (B,1,1)

        noise = tf.random.uniform(observed_data.shape,
                                  dtype=observed_data.dtype)  # noise = torch.randn_like(observed_data).to(self.device)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel.call(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask

        num_eval = tf.reduce_sum(target_mask)
        loss = tf.reduce_sum(residual ** 2) / (num_eval if num_eval > 0 else 1)

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = tf.expand_dims(noisy_data, 1)  # (B,1,K,L)
        else:
            cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
            noisy_target = tf.expand_dims((1 - cond_mask) * noisy_data, 1)
            total_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = tf.zeros([B, n_samples, K, L])  # .to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = tf.random.uniform(noisy_obs.shape, dtype=noisy_obs.dtype)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = tf.random.uniform(observed_data.shape, dtype=observed_data.dtype)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = tf.expand_dims(diff_input, 1)  # (B,1,K,L)
                else:
                    cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
                    noisy_target = tf.expand_dims((1 - cond_mask) * current_sample, 1)
                    diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
                predicted = self.diffmodel.call(diff_input, side_info, tf.constant([t]))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = tf.random.uniform(current_sample.shape, dtype=current_sample.dtype)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample  # .detach()

        return imputed_samples

# TODO Train STEP
    def train_step(self, batch):
        observed_data, observed_mask, gt_mask, for_pattern_mask, cond_mask, time_emb, time_fea = batch[0]
        # observed_tp = tf.range((observed_mask.shape[1],1))
        is_train = 1

        with tf.GradientTape() as tape:
            # tape.watch(learnable_params)
            side_info = self.get_side_info(time_emb, cond_mask, time_fea)
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train)

        learnable_params = (
                self.embed_layer.trainable_variables + self.diffmodel.trainable_variables
        )
        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))
        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # TODO evaluate
    def evaluate(self, batch, n_samples):
        (observed_data, observed_mask, observed_tp, gt_mask, _, cut_length) = self.process_data(batch)
        # with torch.no_grad():
        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        for i in range(len(cut_length)):
            target_mask[i, ..., 0: cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp

    # def process_data(self, batch):
    #     # observed_data = tf.cast(batch["observed_data"], dtype=tf.float32)  # .to(self.device)
    #     # observed_mask = tf.cast(batch["observed_mask"], dtype=tf.float32)
    #     # observed_tp = tf.cast(batch["timepoints"], dtype=tf.float32)
    #     # gt_mask = tf.cast(batch["gt_mask"], dtype=tf.float32)
    #     observed_data, observed_mask, gt_mask, observed_tp = batch
    #
    #     observed_data = tf.transpose(observed_data, perm=[0, 2, 1])
    #     observed_mask = tf.transpose(observed_mask, perm=[0, 2, 1])
    #     gt_mask = tf.transpose(gt_mask, [0, 2, 1])
    #
    #     # cut_length = tf.zeros(observed_data.shape[0], dtype=tf.int64)
    #     for_pattern_mask = observed_mask
    #
    #     return observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask #, cut_length
    #
    #

import einops
import keras

from .CSDI_base import *

''' Standalone CSDI imputer. The imputer class is located in the last part of the notebook, please see more documentation there'''

class tfCSDI(keras.Model):
    def __init__(self, target_dim, config, device):
        super(tfCSDI, self).__init__()
        self.loss_fn = None
        self.optimizer = None
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
        self.embed_layer.add(keras.layers.Input(shape=(self.target_dim,)))
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
        # self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def time_embedding(self, pos, d_model=128):  # pos batch_size * seq_length
        position = tf.cast(tf.expand_dims(pos, 2), dtype=tf.float32)
        pow_y = tf.cast(tf.range(0, d_model, 2) / d_model, dtype=tf.float32)
        div_term = 1 / tf.pow(10000.0, pow_y)
        div_term = tf.cast(div_term, dtype=tf.float32)
        # pe[:, :, 0::2] = tf.math.sin(position * div_term)
        pe_values = tf.stack([tf.math.sin(position * div_term), tf.math.cos(position * div_term)], axis=-1)  # B L 64 2
        pe_values = tf.reshape(pe_values, [tf.shape(pe_values)[0], tf.shape(pe_values)[1],
                                           tf.shape(pe_values)[2] * 2])  # B 100 d_model
        pe = pe_values
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # B d_model L
        time_embed = tf.expand_dims(time_embed, 2)  # B d_model 1 L
        time_embed = tf.tile(time_embed, [1, 1, K, 1])  # B d_model K L
        # input to embed_layer is  self.target_dim, output is self.target_dim * self.emb_feature_dim (14 *16)
        feature = tf.reshape(tf.range(self.target_dim), [1, -1])  # B * target_dim
        feature_embed = tf.expand_dims(self.embed_layer(feature), 0)  # B * target_dim * embed_output_dim
        feature_embed = tf.tile(feature_embed, [tf.shape(cond_mask)[0], L, 1, 1])
        side_info = tf.concat([time_embed, feature_embed], axis=-1)  # (B,L,K,*)
        side_info = rearrange(side_info, 'i j k l -> i l k j')  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask, 1)  # (B,1,K,L)
            side_info = tf.concat([side_info, side_mask], axis=1)

        return side_info

    def compute_loss(self, observed_data, cond_mask, observed_mask, observed_tp, is_train=True, set_t=-1):
        side_info = self.get_side_info(observed_tp, cond_mask)
        if is_train:
            t = tf.random.uniform(shape=(tf.shape(observed_data)[0],), minval=0, maxval=self.num_steps, dtype=tf.int32)
        else:
            t = tf.ones(shape=(tf.shape(observed_data)[0],), dtype=tf.int32) * set_t  #  num_steps (50) * B


        # is_train = tf.constant(is_train, dtype=tf.bool)

        # train = lambda : tf.random.uniform(shape=(tf.shape(observed_data)[0],), minval=0, maxval=self.num_steps, dtype=tf.int32)
        #
        # validate = lambda: tf.ones(shape=(tf.shape(observed_data)[0],), dtype=tf.int32) * set_t # B * num_steps (50)
        #
        # t = tf.cond(
        #     is_train,
        #     true_fn=lambda : tf.random.uniform(shape=(tf.shape(observed_data)[0],), minval=0, maxval=self.num_steps, dtype=tf.int32),
        #     false_fn=lambda: tf.ones(shape=(tf.shape(observed_data)[0],), dtype=tf.int32) * set_t # B * num_steps (50)
        # )
        noise = tf.random.uniform(tf.shape(observed_data), dtype=observed_data.dtype)
        current_alpha = tf.gather(self.alpha_tf, t, axis=0)  # ( (num_steps) * B, 1, 1)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask, is_train)

        predicted = self.diffmodel.__call__(total_input, side_info, t, training=is_train)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        loss = tf.reduce_sum(residual ** 2)
        num_eval = tf.cast(tf.reduce_sum(target_mask), loss.dtype)

        loss = tf.cond(
            num_eval > 0,
            true_fn=lambda: loss/num_eval,
            false_fn=lambda: loss
        )

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask, is_train=True):
        if self.is_unconditional == True:
            total_input = tf.expand_dims(noisy_data, 1)  # (B,1,K,L)
        else:
            cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
            noisy_target = tf.expand_dims((1 - cond_mask) * noisy_data, 1)
            total_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)

        return total_input

    @tf.function
    def impute(self, batch, n_samples):
        observed_data, observed_mask, gt_mask = batch
        cond_mask = gt_mask
        B, K, L = observed_data.shape
        observed_tp = tf.reshape(tf.range(L), [1, L])  # 1 L
        observed_tp = tf.tile(observed_tp, [tf.shape(observed_data)[0], 1])  # B L
        target_mask = observed_mask - cond_mask
        side_info = tf.stop_gradient(
            self.get_side_info(observed_tp, cond_mask)
        )
        # imputed_samples = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False, element_shape=observed_data.shape)
        # i=0

        @tf.function
        def single_sample_imputer(sample_i):
            current_sample = tf.TensorArray(dtype=observed_data.dtype, size=self.num_steps + 1, clear_after_read=False)
            t = self.num_steps - 1
            current_sample = current_sample.write(t + 1, tf.random.uniform(observed_data.shape,
                                                                           dtype=observed_data.dtype))
            while t >= 0:
                # if self.is_unconditional == True:
                #     diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                #     diff_input = tf.expand_dims(diff_input, 1)  # (B,1,K,L)
                # else:
                cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
                noisy_target = tf.expand_dims((1 - cond_mask) * current_sample.read(t + 1), 1)
                diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
                predicted = tf.stop_gradient(
                    self.diffmodel.__call__(diff_input, side_info, tf.constant([t]))
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = current_sample.write(t, coeff1 * (
                        current_sample.read(t + 1) - coeff2 * predicted))  # .mark_used()

                if t > 0:
                    noise = tf.random.uniform(observed_data.shape, dtype=observed_data.dtype)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample = current_sample.write(t, current_sample.read(t) + sigma * noise)  # .mark_used()
                t -= 1
            return current_sample.read(0)
        i = tf.range(n_samples)
        imputed_samples = tf.stop_gradient(tf.map_fn(fn=single_sample_imputer, elems=i,
                                                     fn_output_signature=tf.TensorSpec(shape=observed_data.shape, dtype=observed_data.dtype),
                                                     parallel_iterations=5))
        # while i < n_samples:
        #     current_sample = tf.TensorArray(dtype=observed_data.dtype, size=self.num_steps+1, clear_after_read=False)
        #     t = self.num_steps - 1
        #     current_sample = current_sample.write(t+1, tf.random.uniform(observed_data.shape, dtype=observed_data.dtype))#.mark_used()
        #     while t >= 0:
        #         # if self.is_unconditional == True:
        #         #     diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
        #         #     diff_input = tf.expand_dims(diff_input, 1)  # (B,1,K,L)
        #         # else:
        #         cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
        #         noisy_target = tf.expand_dims((1 - cond_mask) * current_sample.read(t+1), 1)
        #         diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
        #         predicted = tf.stop_gradient(
        #             self.diffmodel.__call__(diff_input, side_info, tf.constant([t]))
        #         )
        #
        #         coeff1 = 1 / self.alpha_hat[t] ** 0.5
        #         coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
        #         current_sample = current_sample.write(t, coeff1 * (current_sample.read(t+1) - coeff2 * predicted))#.mark_used()
        #
        #         if t > 0:
        #             noise = tf.random.uniform(observed_data.shape, dtype=observed_data.dtype)
        #             sigma = (
        #                             (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
        #                     ) ** 0.5
        #             current_sample = current_sample.write(t, current_sample.read(t) + sigma * noise)#.mark_used()
        #         t -= 1
        #     imputed_samples = imputed_samples.write(i, current_sample.read(0))#.mark_used()
        #     i += 1
        # imputed_samples = imputed_samples.stack()
        # outer_cond = lambda i, n_samples: i < n_samples
        # inner_cond = lambda t, current_sample: t >= 0
        #
        # def inner_body(t, current_sample):
        #     # if self.is_unconditional == True:
        #     #     diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
        #     #     diff_input = tf.expand_dims(diff_input, 1)  # (B,1,K,L)
        #     # else:
        #     cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
        #     noisy_target = tf.expand_dims((1 - cond_mask) * current_sample, 1)
        #     diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
        #     predicted = self.diffmodel.__call__(diff_input, side_info, t)
        #
        #     coeff1 = 1 / tf.gather(self.alpha_hat, t) ** 0.5
        #     coeff2 = (1 - tf.gather(self.alpha_hat, t)) / (1 - tf.gather(self.alpha_hat, t)) ** 0.5
        #     current_sample = coeff1 * (current_sample - coeff2 * predicted)
        #
        #     if t > 0:
        #         noise = tf.random.uniform(current_sample.shape, dtype=current_sample.dtype)
        #         sigma = (
        #                         (1.0 - tf.gather(self.alpha, t - 1)) / (1.0 - tf.gather(self.alpha, t)) * tf.gather(self.beta, t)
        #                 ) ** 0.5
        #         current_sample += sigma * noise
        #     return t-1, current_sample
        #
        # def outer_body(i, n_samples):
        #     # generate noisy observation for unconditional model
        #     # if self.is_unconditional == True:
        #     #     noisy_obs = observed_data
        #     #     noisy_cond_history = []
        #     #     t = 0
        #     #     while t < self.num_steps:
        #     #         noise = tf.random.uniform(noisy_obs.shape, dtype=noisy_obs.dtype)
        #     #         noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
        #     #         noisy_cond_history.append(noisy_obs * cond_mask)
        #     #         t += 1
        #     # else:
        #     #     noisy_cond_history = None
        #     current_sample = tf.random.uniform(observed_data.shape, dtype=observed_data.dtype)
        #
        #     t = tf.constant([self.num_steps - 1], dtype=tf.int32)
        #     inner_loop_vals = (t, current_sample)
        #     t, current_sample = tf.while_loop(inner_cond, inner_body, inner_loop_vals, back_prop=False)
        #     imputed_samples.write(tf.reshape(i, []), current_sample)
        #     # return current_sample
        #     return i + 1, n_samples
        #
        # i = tf.constant([0], dtype=tf.int32)
        # _ = tf.while_loop(outer_cond, outer_body, loop_vars=[i, n_samples], back_prop=False)
        # imputed_samples = imputed_samples.stack()

        return imputed_samples, observed_data, target_mask, observed_mask, observed_tp

    def compile(self, optimizer):
        super(tfCSDI, self).compile()
        self.optimizer = optimizer
        self.loss_fn = self.compute_loss

    @tf.function(input_signature=[((tf.TensorSpec([None, 14, 100], tf.float32),
                                    tf.TensorSpec([None, 14, 100], tf.float32),
                                    tf.TensorSpec([None, 14, 100], tf.float32),
                                    tf.TensorSpec([None, 14, 100], tf.float32)),)])
    def train_step(self, batch):
        observed_data, observed_mask, _, cond_mask = batch[0]
        # observation mask denotes the original data missing, gt_masks is manmade mask
        B, K, L = cond_mask.shape
        observed_tp = tf.reshape(tf.range(L), [1, L])
        observed_tp = tf.tile(observed_tp, [tf.shape(observed_data)[0], 1])  # B L

        with tf.GradientTape() as tape:
            loss = self.loss_fn(observed_data, cond_mask, observed_mask, observed_tp)
        learnable_params = (
                self.embed_layer.trainable_variables + self.diffmodel.trainable_variables
        )
        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))
        del gradients
        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # @tf.function
    def test_step(self, batch):
        observed_data, observed_mask, gt_mask, _ = batch[0]
        # observation mask denotes the original data missing, gt_masks is man-made mask
        cond_mask = gt_mask
        B, K, L = observed_data.shape
        observed_tp = tf.reshape(tf.range(L), [1, L])  # 1 L
        observed_tp = tf.tile(observed_tp, [tf.shape(observed_data)[0], 1])  # B, L

        def body(t):
            loss = self.loss_fn(observed_data, cond_mask, observed_mask, observed_tp, is_train=False, set_t=t)
            return loss
        t = tf.range(self.num_steps)
        @tf.function
        def map_fn_no_gradient():
            LOSS_SUM = tf.stop_gradient(tf.map_fn(body, elems=t, parallel_iterations=self.num_steps, fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)))
            return LOSS_SUM
        LOSS_SUM = map_fn_no_gradient()
        val_loss = tf.reduce_sum(LOSS_SUM) / self.num_steps
        self.loss_tracker.update_state(val_loss)
        return {"loss": self.loss_tracker.result()}

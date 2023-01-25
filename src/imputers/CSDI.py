import einops
import keras

from .CSDI_base import *

''' Standalone CSDI imputer. The imputer class is located in the last part of the notebook, please see more documentation there'''


class tfCSDI(keras.Model):
    def __init__(self, target_dim, config):
        super(tfCSDI, self).__init__()
        self.loss_fn = None
        self.optimizer = None
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.algo = config['diffusion']['time_layer']

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        self.embed_layer = keras.layers.Embedding(input_dim=self.target_dim, output_dim=self.emb_feature_dim, trainable=True)

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
        self.alpha = tf.math.cumprod(self.alpha_hat)
        self.alpha_tf = tf.expand_dims(tf.expand_dims(tf.cast(self.alpha, dtype=tf.float32), 1), 1)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compile(self, optimizer):
        super(tfCSDI, self).compile()
        self.optimizer = optimizer
        self.loss_fn = self.compute_loss

    def built_after_run(self):
        self.built = True
        self.embed_layer.built = True
        self.diffmodel.built = True
        self.diffmodel.built_after_run()

    def time_embedding(self, pos, d_model=128):  # pos batch_size * seq_length
        position = tf.cast(tf.expand_dims(pos, 2), dtype=tf.float32)
        pow_y = tf.cast(tf.range(0, d_model, 2) / d_model, dtype=tf.float32)
        div_term = 1 / tf.pow(10000.0, pow_y)
        div_term = tf.cast(div_term, dtype=tf.float32)
        # pe[:, :, 0::2] = tf.math.sin(position * div_term)
        pe_values = tf.stack([tf.math.sin(position * div_term), tf.math.cos(position * div_term)], axis=-1)  # B L 64 2
        pe_values = tf.reshape(pe_values, [tf.shape(pe_values)[0], tf.shape(pe_values)[1], d_model])  # B L 64*2
        return pe_values

    @tf.function
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # B d_model L
        time_embed = tf.expand_dims(time_embed, 2)  # B d_model 1 L
        time_embed = tf.tile(time_embed, [1, 1, K, 1])  # B d_model K L
        # input to embed_layer is  self.target_dim, output is self.target_dim * self.emb_feature_dim (14 *16)
        # TODO: feature embedding has different results compared to torch version
        feature_embed = tf.expand_dims(tf.expand_dims(self.embed_layer(tf.range(self.target_dim)), 0), 0)  # 1 * 1 * target_dim * embed_output_dim
        feature_embed = tf.tile(feature_embed, [tf.shape(cond_mask)[0], L, 1, 1])
        side_info = tf.concat([time_embed, feature_embed], axis=-1)  # (B,L,K,*)
        side_info = rearrange(side_info, 'i j k l -> i l k j')  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask, 1)  # (B,1,K,L)
            side_info = tf.concat([side_info, side_mask], axis=1)

        return side_info

    @tf.function
    def call(self, inputs):
        total_input, observed_tp, cond_mask, t = inputs
        side_info = self.get_side_info(observed_tp, cond_mask)
        diff_input_batch = (total_input, side_info, t)
        predicted = self.diffmodel(diff_input_batch)  # (B,K,L)
        return predicted

    @tf.function
    def compute_loss(self, observed_data, cond_mask, observed_mask, observed_tp, is_train=True, set_t=-1):
        # side_info = self.get_side_info(observed_tp, cond_mask)
        # if is_train:
        #     t = tf.random.uniform(shape=(tf.shape(observed_data)[0],), minval=0, maxval=self.num_steps, dtype=tf.int32)
        # else:
        #     t = tf.ones(shape=(tf.shape(observed_data)[0],), dtype=tf.int32) * set_t  #  num_steps (50) * B
        t = tf.cond(
            tf.constant(is_train),
            true_fn=lambda : tf.random.uniform(shape=(tf.shape(observed_data)[0],), minval=0, maxval=self.num_steps, dtype=tf.int32),
            false_fn=lambda : tf.ones(shape=(tf.shape(observed_data)[0],), dtype=tf.int32) * set_t
        )
        noise = tf.random.normal(tf.shape(observed_data), dtype=observed_data.dtype)
        current_alpha = tf.gather(self.alpha_tf, t, axis=0)  # (B, 1, 1)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask, is_train)

        predicted = self((total_input, observed_tp, cond_mask, t)) # se

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

    @tf.function(input_signature=[((tf.TensorSpec([None, 6, 29], tf.float32),
                                    tf.TensorSpec([None, 6, 29], tf.float32),
                                    tf.TensorSpec([None, 6, 29], tf.float32),
                                    tf.TensorSpec([None, 6, 29], tf.float32)), )])
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

    @tf.function(input_signature=[((tf.TensorSpec([None, 6, 29], tf.float32),
                                    tf.TensorSpec([None, 6, 29], tf.float32),
                                    tf.TensorSpec([None, 6, 29], tf.float32),
                                    tf.TensorSpec([None, 6, 29], tf.float32)), )])
    def test_step(self, batch):
        observed_data, observed_mask, _, cond_mask = batch[0]
        # observation mask denotes the original data missing, gt_masks is man-made mask
        B, K, L = observed_data.shape
        observed_tp = tf.reshape(tf.range(L), [1, L])  # 1 L
        observed_tp = tf.tile(observed_tp, [tf.shape(observed_data)[0], 1])  # B, L

        def body(t):
            loss = self.loss_fn(observed_data, cond_mask, observed_mask, observed_tp, is_train=False, set_t=t)
            return tf.stop_gradient(loss)
        t = tf.range(self.num_steps)
        # try:
        #     LOSS_SUM = tf.vectorized_map(body, elems=t) #, parallel_iterations=10,
        # except:
        LOSS_SUM = tf.map_fn(body, elems=t, fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32))
        val_loss = tf.reduce_sum(LOSS_SUM) / self.num_steps
        self.loss_tracker.update_state(val_loss)
        return {"loss": self.loss_tracker.result()}

    # @tf.function(input_signature=[(tf.TensorSpec([None, 14, 100], tf.float32),
    #                                 tf.TensorSpec([None, 14, 100], tf.float32),
    #                                 tf.TensorSpec([None, 14, 100], tf.float32)),
    #                                tf.TensorSpec([], tf.int32)])
    # def impute(self, batch, n_samples):
    #     observed_data, observed_mask, gt_mask = batch
    #     cond_mask = gt_mask
    #     B, K, L = observed_data.shape
    #     observed_tp = tf.reshape(tf.range(L), [1, L])  # 1 L
    #     observed_tp = tf.tile(observed_tp, [tf.shape(observed_data)[0], 1])  # B L
    #     target_mask = observed_mask - cond_mask
    #     side_info = tf.stop_gradient(
    #         self.get_side_info(observed_tp, cond_mask)
    #     )
    #
    #     # @tf.function
    #     # def single_sample_imputer(sample_i):
    #     imputed_samples = tf.TensorArray(dtype=tf.float32, size=n_samples)
    #     sample_i = 0
    #     while sample_i < n_samples:
    #         current_sample = tf.TensorArray(dtype=tf.float32, size=1, clear_after_read=False)
    #         t = self.num_steps - 1
    #         current_sample = current_sample.write(0, tf.random.normal(observed_data.shape, dtype=observed_data.dtype))
    #         while t >= 0:
    #             # if self.is_unconditional == True:
    #             #     diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
    #             #     diff_input = tf.expand_dims(diff_input, 1)  # (B,1,K,L)
    #             # else:
    #             cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
    #             noisy_target = tf.expand_dims((1 - cond_mask) * current_sample.read(0), 1)
    #             diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
    #             predicted = tf.stop_gradient(
    #                 self.diffmodel(diff_input, side_info, tf.constant([t]))
    #             )
    #
    #             coeff1 = 1 / self.alpha_hat[t] ** 0.5
    #             coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
    #             current_sample = current_sample.write(0, coeff1 * (current_sample.read(0) - coeff2 * predicted))
    #
    #             if t > 0:
    #                 noise = tf.random.normal(observed_data.shape, dtype=observed_data.dtype)
    #                 sigma = (
    #                                 (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
    #                         ) ** 0.5
    #                 current_sample = current_sample.write(0, current_sample.read(0) + sigma * noise)  # .mark_used()
    #             t -= 1
    #         imputed_samples = imputed_samples.write(sample_i, current_sample.read(0))
    #         sample_i += 1
    #     imputed_samples = imputed_samples.stack()
    #     return imputed_samples, observed_data, target_mask, observed_mask, observed_tp
    #

import numpy as np
import random
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import math
import argparse
import datetime
import json
import yaml
import os
# from torch.utils.data import DataLoader, Dataset


''' Standalone CSDI imputer. The imputer class is located in the last part of the notebook, please see more documentation there'''



def train(model, config, X, y, valid_loader=None, valid_epoch_interval=50, path_save="", log_dir=''):

    output_path = f"{path_save}-model"
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    boundaries = [p1, p2]
    values = [config["lr"], config["lr"] * 0.1, config["lr"] * 0.1 * 0.1]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-6)
    loss =  keras.losses.MeanSquaredError()

    best_valid_loss = 1e10
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    earlyStop_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)
    earlyStop_accu_call_back = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', patience=3)
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path_save,
        save_weights_only=False,
        monitor='accuracy',
        mode='max',
        save_best_only=True,
    )

    # training
    # net.build(input_shape=((model_config["in_channels"], None, ), (model_config["in_channels"], None, ), (model_config["in_channels"], None, ), (None,)))
    # net.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(
        x=X,
        y=y,
        batch_size=64,
        epochs=1000,
        callbacks=[tensorboard_callback,
                   earlyStop_loss_callback,
                   earlyStop_accu_call_back,
                   best_checkpoint_callback],
    )
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        with tqdm(train_loader, mininterval=5.0, maxinterval=5.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_variables)
                    loss = model(train_batch)
                grad = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grad, model.trainable_variables))
                avg_loss += loss
                del grad
                it.set_postfix(
                    ordered_dict={"avg_epoch_loss": avg_loss / batch_no,"epoch": epoch_no + 1},refresh=False)
        model.save(output_path)
# TODO evaluate in keras
        # if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
        #     model.eval()
        #     avg_loss_valid = 0
        #     with torch.no_grad():
        #         with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
        #             for batch_no, valid_batch in enumerate(it, start=1):
        #                 loss = model(valid_batch, is_train=0)
        #                 avg_loss_valid += loss.item()
        #                 it.set_postfix(ordered_dict={"valid_avg_epoch_loss":avg_loss_valid/batch_no,"epoch":epoch_no},refresh=False)
        #     if best_valid_loss > avg_loss_valid:
        #         best_valid_loss = avg_loss_valid
        #         print("\n best loss is updated to ",avg_loss_valid/batch_no,"at",epoch_no+1)
        # try:
        #   wandb.log({"loss_valid": avg_loss_valid / batch_no})
        # except:
        #   pass

    
def quantile_loss(target, forecast, q: tf.float32, eval_points) -> tf.float32:
    return 2 * tf.reduce_sum(tf.math.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)))


def calc_denominator(target, eval_points):
    return tf.reduce_sum(tf.math.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    num_quantiles = quantiles.shape[0]
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            # TODO quantile
            q_pred.append(np.quantile(forecast[j: j + 1], quantiles[i], axis=1)) # use numpy quantile here
        q_pred = tf.concat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
        
    return CRPS / num_quantiles

# TODO
def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, path_save=""):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=5.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no}, refresh=True)
                
            with open(f"{path_save}generated_outputs_nsample"+str(nsample)+".pk","wb") as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

            with open(f"{path_save}result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total, 
                        CRPS
                    ], 
                    f)
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)


    return all_generated_samples.cpu().numpy()

# TODO transformer keras
def transformer_encoder(inputs, head_size, num_heads, ff_dim, activation, dropout=0):
    # Normalization and Attention
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation, data_format="channels_first")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, data_format="channels_first")(x)
    return x + res

def TransformerEncoder(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    activation,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, activation, dropout)

    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_torch_trans(heads=8, layers=1, channels=64):
    return TransformerEncoder((channels, ),
                              head_size=channels,
                              num_heads=heads,
                              ff_dim=64,
                              num_transformer_blocks=layers,
                              mlp_units=[],
                              activation='gelu',
                              )
    # encoder_layer = nn.TransformerEncoderLayer(
    #     d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    # return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size, initializer=None, activation=None):
    if initializer is None:
        initializer = tf.keras.initializers.HeNormal()
    layer = keras.layers.Conv1D(out_channels, kernel_size, data_format="channels_first", kernel_initializer=initializer, activation=activation)
    return layer

def swish(x):
    return x * keras.activations.sigmoid(x)


def _build_embedding(num_steps, dim=64):
    steps = tf.expand_dims(tf.range(num_steps), 1)  # (T,1)
    frequencies = 10.0 ** tf.expand_dims(tf.range(dim) / (dim - 1) * 4.0, 0)  # (1,dim)
    table = steps * frequencies  # (T,dim)
    table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
    return table


class DiffusionEmbedding(keras.Model):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        # TODO persistent?
        setattr(self, "embedding", tf.Variable(_build_embedding(num_steps, int(embedding_dim / 2)), name="embedding", trainable=False))
        self.projection = keras.Sequential()
        self.projection.add(keras.layers.Dense(projection_dim, activation=swish))
        self.projection.add(keras.layers.Dense(projection_dim, activation=swish))

    def call(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection(x)
        return x


class diff_CSDI(keras.Model):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1, activation='relu')
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1, activation='relu')
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1,  tf.keras.initializers.Zeros())

        self.residual_layers =[]
        for _ in range(config["layers"]):
            self.residual_layers.append(
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
            )

    def call(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = tf.reshape(x, [B, inputdim, K * L])
        x = self.input_projection(x)
        x = tf.reshape(x, [B, self.channels, K, L])

        diffusion_emb = self.diffusion_embedding.call(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer.all(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = tf.reduce_sum(tf.stack(skip), axis=0) / tf.math.sqrt(len(self.residual_layers))
        x = tf.reshape(x, [B, self.channels, K * L])
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = tf.reshape(x,[B, K, L])
        return x

    
class ResidualBlock(keras.Model):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = keras.layers.Dense(channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = tf.transpose(tf.reshape(y,[B, channel, K, L]), [0, 2, 1, 3])
        y = tf.reshape(y, [B * K, channel, L])
        y = self.time_layer(tf.transpose(y, [2, 0, 1]))
        y = tf.transpose(y, [1, 2, 0])
        y = tf.transpose(tf.reshape(y, [B, K, channel, L]), [0, 2, 1, 3])
        y = tf.reshape(y, [B, channel, K * L])
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = tf.transpose(tf.reshape(y, [B, channel, K, L]), [0, 3, 1, 2])
        y = tf.reshape(y, [B * L, channel, K])
        y = self.feature_layer(tf.transpose(y, [2, 0, 1]))
        y = tf.transpose(y, [1, 2, 0])
        y = tf.transpose(tf.reshape(y, [B, L, channel, K]), [0, 2, 3, 1])
        y = tf.reshape(y, [B, channel, K * L])
        return y

    def call(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = tf.reshape(x, [B, channel, K * L])

        diffusion_emb = self.diffusion_projection(diffusion_emb)
        diffusion_emb = tf.expand_dims(diffusion_emb, -1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = tf.reshape(cond_info, [B, cond_dim, K * L])
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        
        gate, filter = tf.split(y, 2, axis=1)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = tf.split(y, 2, axis=1)
        x = tf.reshape(x, base_shape)
        residual = tf.reshape(residual, base_shape)
        skip = tf.reshape(skip, base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CSDI_base(keras.Model):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = keras.layers.Embedding(inout_dim=self.target_dim, output_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = tf.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = tf.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = tf.math.cumprod(self.alpha_hat) # TODO numpy default is flattened need to check shape
        self.alpha_tf = tf.expand_dims(tf.expand_dims(tf.cast(self.alpha, dtype=tf.float32), 1), 1)

    def time_embedding(self, pos, d_model=128):
        pe = tf.zeros(pos.shape[0], pos.shape[1], d_model) #.to(self.device)
        position = tf.expand_dims(pos, 2)
        div_term = 1 / tf.pow(10000.0, tf.range(0, d_model, 2) / d_model) # .to(self.device)
        pe[:, :, 0::2] = tf.math.sin(position * div_term)
        pe[:, :, 1::2] = tf.math.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = np.random.uniform(observed_mask.shape) * observed_mask
        rand_for_mask = np.reshape(rand_for_mask, [len(rand_for_mask), -1])
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  
            num_observed = (observed_mask[i]).sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = tf.reshape(tf.convert_to_tensor(rand_for_mask > 0), observed_mask.shape)
        cond_mask = tf.cast(cond_mask, dtype=tf.float32)
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
# TODO tensor assignment
        cond_mask = tf.identity(observed_mask)
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else: 
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    
    
    def get_side_info(self, observed_tp, cond_mask):
        
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = tf.expand_dims(time_embed, 2) #.expand(-1, -1, K, -1)
        time_embed = tf.tile(time_embed, [1,1,K,1])
        feature_embed = self.embed_layer(tf.range(self.target_dim)) #.to(self.device))  # (K,emb)
        feature_embed = tf.expand_dims(feature_embed, 0) # .unsqueeze(0).expand(B, L, -1, -1)
        feature_embed = tf.tile(tf.expand_dims(feature_embed, 0), [B, L, 1, 1])
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
            loss_sum += loss #.detach()
            
        return loss_sum / self.num_steps

    
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = tf.cast(tf.ones(B) * set_t, dtype=tf.int64) #.long().to(self.device)
        else:
            # t = torch.randint(0, self.num_steps, [B]).to(self.device)
            t = tf.random.uniform(shape=(B,), minval=0, maxval=self.num_steps + 1, dtype=tf.int32)
        current_alpha = self.alpha_torch[t]  # (B,1,1)

        noise = tf.random.uniform(observed_data.shape, dtype=observed_data.dtype) # noise = torch.randn_like(observed_data).to(self.device)
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
        imputed_samples = tf.zeros([B, n_samples, K, L]) #.to(self.device)

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
            imputed_samples[:, i] = current_sample #.detach()
            
        return imputed_samples

    
    def call(self, batch, is_train=1):
        (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,_) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)
# TODO evaluate
    def evaluate(self, batch, n_samples):
        (observed_data,observed_mask,observed_tp,gt_mask,_,cut_length) = self.process_data(batch)
        # with torch.no_grad():
        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        for i in range(len(cut_length)):
            target_mask[i, ..., 0: cut_length[i].item()] = 0
                
        return samples, observed_data, target_mask, observed_mask, observed_tp

    
class CSDI_Custom(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Custom, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = tf.cast(batch["observed_data"], dtype=tf.float32) #.to(self.device)
        observed_mask = tf.cast(batch["observed_mask"], dtype=tf.float32)
        observed_tp = tf.cast(batch["timepoints"], dtype=tf.float32)
        gt_mask = tf.cast(batch["gt_mask"], dtype=tf.float32)

        observed_data = tf.transpose(observed_data, perm=[0, 2, 1])
        observed_mask = tf.transpose(observed_mask, perm=[0, 2, 1])
        gt_mask = tf.transpose(gt_mask, [0, 2, 1])

        cut_length = tf.zeros(len(observed_data), dtype=tf.int64)
        for_pattern_mask = observed_mask

        return (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,cut_length)
    
    
def mask_missing_train_rm(data, missing_ratio=0.0):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_nrm(data, k_segments=5):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)

    for channel in range(gt_masks.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_bm(data, k_segments=5):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)
    s_nan = random.choice(list_of_segments_index)

    for channel in range(gt_masks.shape[1]):
        gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_impute(data, mask):
    
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    mask = mask.astype("float32")
    gt_masks = observed_masks * mask

    return observed_values, observed_masks, gt_masks

# TODO Dataset class
class Custom_Train_Dataset(tf.data.Dataset):
    def __init__(self, series, path_save, variant_tensor, use_index_list=None, missing_ratio_or_k=0.0, masking='rm',
                 ms=None):
        super().__init__(variant_tensor)
        self.series = series
        self.length = series.shape[1]
        self.n_channels = series.shape[2]

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        if not os.path.isfile(path_save):  # if datasetfile is none, create
            for sample in series:
                if masking == 'rm':
                    sample = sample.numpy()
                    observed_values, observed_masks, gt_masks = mask_missing_train_rm(sample, missing_ratio_or_k)
                    observed_values = tf.convert_to_tensor(observed_values)
                    observed_masks = tf.convert_to_tensor(observed_masks)
                    gt_masks = tf.convert_to_tensor(gt_masks)
                elif masking == 'nrm':
                    sample = sample.numpy()
                    observed_values, observed_masks, gt_masks = mask_missing_train_nrm(sample, missing_ratio_or_k)
                    observed_values = tf.convert_to_tensor(observed_values)
                    observed_masks = tf.convert_to_tensor(observed_masks)
                    gt_masks = tf.convert_to_tensor(gt_masks)
                elif masking == 'bm':
                    sample = sample.numpy()
                    observed_values, observed_masks, gt_masks = mask_missing_train_bm(sample, missing_ratio_or_k)
                    observed_values = tf.convert_to_tensor(observed_values)
                    observed_masks = tf.convert_to_tensor(observed_masks)
                    gt_masks = tf.convert_to_tensor(gt_masks)
                    
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
                
        
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)

    
class Custom_Impute_Dataset(tf.data.Dataset):
    def __init__(self, series, mask, variant_tensor, use_index_list=None, path_save=''):
        super().__init__(variant_tensor)
        self.series = series
        self.n_channels = series.shape[2]
        self.length = series.shape[1]
        self.mask = mask 

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        if not os.path.isfile(path_save):  # if datasetfile is none, create
            for sample in series:
                
                sample = sample.numpy()
                observed_masks = sample.copy()
                observed_masks[observed_masks!=0] = 1 
                gt_masks = mask
                
                #observed_values, observed_masks, gt_masks = mask_missing_impute(sample, mask)
                
                self.observed_values.append(sample)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)

                
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
    
    
def get_dataloader_train_impute(series,
                                batch_size=4,
                                missing_ratio_or_k=0.1,
                                train_split=0.7,
                                valid_split=0.9,
                                len_dataset=100,
                                masking='rm',
                               path_save='',
                               ms=None):
    indlist = np.arange(len_dataset)

    tr_i, v_i, te_i = np.split(indlist,
                               [int(len(indlist) * train_split),
                                int(len(indlist) * (train_split + valid_split))])

    train_dataset = Custom_Train_Dataset(series=series, use_index_list=tr_i,
                                         missing_ratio_or_k=missing_ratio_or_k, 
                                         masking=masking, path_save=path_save, ms=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Custom_Train_Dataset(series=series, use_index_list=v_i, 
                                         missing_ratio_or_k=missing_ratio_or_k, 
                                         masking=masking, path_save=path_save)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Custom_Train_Dataset(series=series, use_index_list=te_i, 
                                        missing_ratio_or_k=missing_ratio_or_k, 
                                        masking=masking, path_save=path_save)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def get_dataloader_impute(series, mask, batch_size=4, len_dataset=100):
    indlist = np.arange(len_dataset)
    impute_dataset = Custom_Impute_Dataset(series=series, use_index_list=indlist,mask=mask)
    impute_loader = DataLoader(impute_dataset, batch_size=batch_size, shuffle=False)

    return impute_loader



class CSDIImputer:
    def __init__(self, device):
        np.random.seed(0)
        random.seed(0)
        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        '''
        CSDI imputer
        3 main functions:
        a) training based on random missing, non-random missing, and blackout masking.
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              series,
              masking ='rm',
              missing_ratio_or_k = 0.0,
              train_split = 0.7,
              valid_split = 0.2,
              epochs = 200,
              samples_generate = 10,
              path_save = "",
              batch_size = 16,
              lr = 1.0e-3,
              layers = 4,
              channels = 64,
              nheads = 8,
              difussion_embedding_dim = 128,
              beta_start = 0.0001,
              beta_end = 0.5,
              num_steps = 50,
              schedule = 'quad',
              is_unconditional = 0,
              timeemb = 128,
              featureemb = 16,
              target_strategy = 'random',
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
        config['train']['path_save'] = path_save
        
       
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

        config_filename = path_save + "config_csdi_training"
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(config, f, indent=4)


        train_loader, valid_loader, test_loader = get_dataloader_train_impute(
            series=series,
            train_split=config["train"]["train_split"],
            valid_split=config["train"]["valid_split"],
            len_dataset=series.shape[0],
            batch_size=config["train"]["batch_size"],
            missing_ratio_or_k=config["model"]["missing_ratio_or_k"],
            masking=config['model']['masking'],
            path_save=config['train']['path_save'])

        model = CSDI_Custom(config, self.device, target_dim=series.shape[2]) #.to(self.device)
# TODO keras compile fit and evaluate
        train(model=model,
              config=config["train"],
              train_loader=train_loader,
              valid_loader=valid_loader,
              path_save=config['train']['path_save'])

        evaluate(model=model,
                 test_loader=test_loader,
                 nsample=samples_generate,
                 scaler=1,
                 path_save=config['train']['path_save'])
        
        
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
    

    def impute(self,
               sample,
               mask,
               device,
               n_samples = 50,

               ):
        
        '''
        Imputation function 
        sample: sample(s) to be imputed (Samples, Length, Channel)
        mask: mask where values to be imputed. 0's to impute, 1's to remain. 
        n_samples: number of samples to be generated
        return imputations with shape (Samples, N imputed samples, Length, Channel)
        '''
        
        if len(sample.shape) == 2:
            self.series_impute = tf.convert_to_tensor(np.expand_dims(sample, axis=0))
        elif len(sample.shape) == 3:
            self.series_impute = sample

        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.path_config, "r") as f:
            config = json.load(f)

        test_loader = get_dataloader_impute(series=self.series_impute,len_dataset=len(self.series_impute),
                                            mask=mask, batch_size=config['train']['batch_size'])

        model = CSDI_Custom(config, self.device, target_dim=self.series_impute.shape[2]) #.to(self.device)

        # model.load_state_dict(torch.load((self.path_load_model_dic)))
        model.load(self.path_load_model_dic)

        imputations = evaluate(model=model,
                                test_loader=test_loader,
                                nsample=n_samples,
                                scaler=1,
                                path_save='')
        
        indx_imputation = tf.cast(~mask, tf.bool)
            
        original_sample_replaced =[]
        
        for original_sample, single_n_samples in zip(sample.numpy(), imputations): # [x,x,x] -> [x,x] & [x,x,x,x] -> [x,x,x]
            single_sample_replaced = []
            for sample_generated in single_n_samples:  # [x,x] & [x,x,x] -> [x,x]
                sample_out = original_sample.copy()                         
                sample_out[indx_imputation] = sample_generated[indx_imputation]
                single_sample_replaced.append(sample_out)
            original_sample_replaced.append(single_sample_replaced)
            
        output = np.array(original_sample_replaced)
        
        
        return output


    



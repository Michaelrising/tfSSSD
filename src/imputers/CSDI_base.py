import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import math
from einops import rearrange
from .Encoder_keras import Encoder
from .S4Model import S4Layer


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
            q_pred.append(np.quantile(forecast[j: j + 1], quantiles[i], axis=1))  # use numpy quantile here
        q_pred = tf.concat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom

    return CRPS / num_quantiles


def mask_missing_train_rm(data, missing_ratio=0.0):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values) # observed masks: 1: not missing, 0: missing

    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist() # observed data's indices(non-missing)
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    # generated missing indices from the not missing indices
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)
    # gt_masks has two parts: the generated missing masks and the true missing masks
    # reflecting the pattern of the true missing distribution
    observed_values = np.nan_to_num(observed_values) # nan to 0
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


def TrainDataset(series, missing_ratio_or_k=0.0, masking='rm'):
    length = series.shape[1]
    observed_values_list = []
    observed_masks_list = []
    gt_masks_list = []

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

        observed_values_list.append(observed_values)
        observed_masks_list.append(observed_masks)
        gt_masks_list.append(gt_masks)
    observed_values_tensor = tf.stack(observed_values_list)
    observed_masks_tensor = tf.stack(observed_masks_list)
    gt_mask_tensor = tf.stack(gt_masks_list)
    return [observed_values_tensor, observed_masks_tensor, gt_mask_tensor]


def ImputeDataset(series, mask):
    n_channels = series.shape[2]
    length = series.shape[1]
    mask = mask

    observed_values_list = []
    observed_masks_list = []
    gt_masks_list = []

    for sample in series:
        sample = sample.numpy()
        observed_masks = sample.copy()
        observed_masks[observed_masks != 0] = 1
        gt_masks = mask

        # observed_values, observed_masks, gt_masks = mask_missing_impute(sample, mask)

        observed_values_list.append(sample)
        observed_masks_list.append(observed_masks)
        gt_masks_list.append(gt_masks)

    observed_values_tensor = tf.convert_to_tensor(np.stack(observed_values_list))
    observed_masks_tensor = tf.convert_to_tensor(np.stack(observed_masks_list))
    gt_masks_tensor = tf.convert_to_tensor(np.stack(gt_masks_list))

    return [observed_values_tensor, observed_masks_tensor, gt_masks_tensor]


def get_torch_trans(heads=8, layers=1, in_channels=64, out_channels=64):
    return Encoder(num_layers=layers, d_model=out_channels, num_heads=heads, dff=64)


def Conv1d_with_init(in_channels, out_channels, kernel_size, initializer=None, activation=None):
    if initializer is None:
        initializer = tf.keras.initializers.HeNormal()
    layer = keras.layers.Conv1D(out_channels, kernel_size, data_format="channels_first", kernel_initializer=initializer,
                                activation=activation)
    return layer


def swish(x):
    return x * keras.activations.sigmoid(x)


class DiffusionEmbedding(keras.Model):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        # # TODO persistent?
        setattr(self, "embedding",
                tf.Variable(self._build_embedding(num_steps, int(embedding_dim / 2)), name="embedding", trainable=False))
        self.projection = keras.Sequential()
        self.projection.add(keras.layers.Dense(projection_dim, activation=swish))
        self.projection.add(keras.layers.Dense(projection_dim, activation=swish))

    def call(self, t, training=True):
        x = tf.gather(self.embedding, t)
        x = self.projection(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = tf.expand_dims(tf.range(num_steps), 1)  # (T,1)
        frequencies = 10.0 ** tf.expand_dims(tf.range(dim) / (dim - 1) * 4.0, 0)  # (1,dim)
        steps = tf.cast(steps, tf.float32)
        frequencies = tf.cast(frequencies, tf.float32)
        table = steps * frequencies  # (T,dim)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
        return table


class diff_CSDI(keras.Model):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        # feature first
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1, activation='relu')

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1, activation='relu')
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1, tf.keras.initializers.Zeros())

        # feature last
        self.residual_layers = []
        for _ in range(config["layers"]):
            self.residual_layers.append(
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    time_layer=config['time_layer']
                )
            )

    # @tf.function
    def __call__(self, x, cond_info, t, training=True):
        B, inputdim, K, L = x.shape
        x = rearrange(x, '... k l -> ... (k l)')
        # x = tf.reshape(x, [B, inputdim, K * L])
        x = self.input_projection(x)
        x = rearrange(x, '... (k l) -> ... k l', k=K)
        diffusion_emb = self.diffusion_embedding.call(t, training=training)

        skip = tf.TensorArray(dtype=tf.float32, size=len(self.residual_layers))
        for i, layer in enumerate(self.residual_layers):
            x, skip_connection = layer.call(x, cond_info, diffusion_emb, training=training)
            skip = skip.write(i, skip_connection)

        x = tf.reduce_sum(skip.stack(), axis=0) / tf.math.sqrt(float(len(self.residual_layers)))
        # x = tf.reshape(x, [B, self.channels, K * L])
        x = rearrange(x, '... k l -> ... (k l)')
        x = self.output_projection1(x)  # (... B,channel,K*L)
        x = self.output_projection2(x)  # (... B,1,K*L)
        # x = tf.reshape(x, [B, K, L])
        x = rearrange(tf.squeeze(x, 1), '... (k l) -> ... k l', k=K)
        return x


class ResidualBlock(keras.Model):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, time_layer='transformer'):
        super().__init__()
        self.time_layer_type=time_layer
        self.diffusion_projection = keras.layers.Dense(channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = keras.Sequential()
        if time_layer=='transformer':
            self.time_layer.add(keras.layers.Input((None, channels)))
            self.time_layer.add(get_torch_trans(heads=nheads, layers=1, in_channels=channels, out_channels=channels))
        else:
            # self.time_layer.add(keras.layers.Input((channels, None)))
            self.time_layer.add(S4Layer(features=channels, lmax=100))
        self.feature_layer = keras.Sequential()
        self.feature_layer.add(keras.layers.Input((None, channels)))
        self.feature_layer.add(get_torch_trans(heads=nheads, layers=1, in_channels=channels, out_channels=channels))

    def forward_time(self, y, base_shape, training=True):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = rearrange(y, '... c (k l) -> ... k c l', k=K) # b k c l
        if self.time_layer_type=='transformer':
            y = rearrange(y, ' b k c l -> l (b k) c') # in torch version, batch_first is False so it transposes input as L B C but we dont need to do here
            y = self.time_layer(y)
            y = tf.transpose(y, [1, 2, 0])
            y = rearrange(y, '(b k) c l -> b c (k l)', k=K)  # b c k l
        else:
            y = rearrange(y, ' b k c l -> (b k) c l')
            y = self.time_layer(y)  # bk, c, l -> bk, l, c
            y = rearrange(y, '(b k) l c -> b c (k l)', k=K)  # b c k l

        return y

    def forward_feature(self, y, base_shape, training=True):
        B, channel, K, L = base_shape

        if K == 1:
            return y
        y = rearrange(y, 'b c (k l) ->  (b l) c k', k=K) # (bl) c k
        y = rearrange(y, '(b l) c k -> k (b l) c', l=L)
        y = self.feature_layer(y) # k (b l) c
        y = rearrange(y, ' k (b l) c ->  (b l) c k', l=L) # (b l) c k
        y = rearrange(y, ' (b l) c k -> b c k l', l=L) # b c k l
        y = rearrange(y, ' b c k l -> b c (k l)', l=L)
        return y

    def call(self, x, cond_info, diffusion_emb, training=True):
        B, channel, K, L = x.shape
        _, cond_dim, _, _ = cond_info.shape
        base_shape = x.shape
        x = rearrange(x, '... k l -> ... (k l)')

        diffusion_emb = self.diffusion_projection(diffusion_emb)
        diffusion_emb = tf.expand_dims(diffusion_emb, -1)  # ((n_steps), B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, training=training)
        y = self.forward_feature(y, base_shape, training=training)  # (... B,channel,K*L)
        y = self.mid_projection(y)  # (... B,2*channel,K*L)


        cond_info = rearrange(cond_info, '... k l -> ... (k l)')
        cond_info = self.cond_projection(cond_info)  # (... B, 2*channel,K*L)
        y = y + cond_info

        gate, filter = tf.split(y, 2, axis=1)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (... B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = tf.split(y, 2, axis=1)
        x = rearrange(x, '... (k l) -> ... k l ', k=K)
        residual = rearrange(residual, '... (k l) -> ... k l ', k=K)
        skip = rearrange(skip, '... (k l) -> ... k l ', k=K)
        return (x + residual) / math.sqrt(2.0), skip

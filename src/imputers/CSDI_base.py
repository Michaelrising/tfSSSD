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
    # timepoints = tf.reshape(tf.convert_to_tensor(np.arange(length)), )

    # data_dict = {"observed_data": observed_values_tensor, "observed_mask": observed_masks_tensor,
    #              "gt_mask": gt_mask_tensor} #, "timepoints": timepoints}

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
    data_dict = {"observed_data": observed_values_tensor, "observed_mask": observed_masks_tensor,
                 "gt_mask": gt_masks_tensor}

    return data_dict


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
    x = keras.layers.Dense(inputs.shape[-1])(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(inputs.shape[-1])(x)
    return x + res


def TransformerEncoder(
        input_shape,
        out_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        activation,
        dropout=0,
        mlp_dropout=0,
):
    inputs = keras.Input(shape=(None, input_shape,)) # (batch size, sequence length, features)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, activation, dropout)

    x = keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(out_shape, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_torch_trans(heads=8, layers=1, in_channels=64, out_channels=64):
    return TransformerEncoder(in_channels,
                              out_channels,
                              head_size=in_channels,
                              num_heads=heads,
                              ff_dim=64,
                              num_transformer_blocks=layers,
                              mlp_units=[],
                              activation='gelu',
                              )


def Conv1d_with_init(in_channels, out_channels, kernel_size, initializer=None, activation=None):
    if initializer is None:
        initializer = tf.keras.initializers.HeNormal()
    layer = keras.layers.Conv1D(out_channels, kernel_size, data_format="channels_first", kernel_initializer=initializer,
                                activation=activation)
    return layer


def swish(x):
    return x * keras.activations.sigmoid(x)


def _build_embedding(num_steps, dim=64):
    steps = tf.expand_dims(tf.range(num_steps), 1)  # (T,1)
    frequencies = 10.0 ** tf.expand_dims(tf.range(dim) / (dim - 1) * 4.0, 0)  # (1,dim)
    steps = tf.cast(steps, tf.float32)
    frequencies = tf.cast(frequencies, tf.float32)
    table = steps * frequencies  # (T,dim)
    table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
    return table


class DiffusionEmbedding(keras.Model):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        # TODO persistent?
        setattr(self, "embedding",
                tf.Variable(_build_embedding(num_steps, int(embedding_dim / 2)), name="embedding", trainable=False))
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

        # feature first
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1, activation='relu')
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
        x = tf.reshape(x, [B, K, L])
        return x


class ResidualBlock(keras.Model):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = keras.layers.Dense(channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, in_channels=channels, out_channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, in_channels=channels, out_channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = tf.transpose(tf.reshape(y, [B, channel, K, L]), [0, 2, 1, 3])
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

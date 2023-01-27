import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import math
from einops import rearrange
# from .Encoder_keras import Encoder
import tensorflow_models as tfm

from .S4Model import S4Layer
from .S5Model import S5Layer
from .MegaModel import Mega, MegaLayer
# from src.utils.util import SetLearningRate


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


# def mask_missing_train_holiday(data, holiday):
#     observed_values = np.array(data)
#     observed_masks = ~np.isnan(observed_values)
#     gt_masks = observed_masks.copy()
#     length_index = np.array(range(data.shape[0]))
#     list_of_segments_index = np.array_split(length_index, k_segments)
#     s_nan = random.choice(list_of_segments_index)
#
#     for channel in range(gt_masks.shape[1]):
#         gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0
#
#     observed_values = np.nan_to_num(observed_values)
#     observed_masks = observed_masks.astype("float32")
#     gt_masks = gt_masks.astype("float32")
#
#     return observed_values, observed_masks, gt_masks
#

def mask_missing_impute(data, mask):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    mask = mask.astype("float32")
    gt_masks = observed_masks * mask

    return observed_values, observed_masks, gt_masks


def TrainDataset(series, missing_ratio_or_k=0.0, masking='rm', batch_size=None):
    B, L, K = series.shape
    observed_values_list = []
    observed_masks_list = []
    gt_masks_list = []
    if masking == 'holiday':
        series = series.numpy()
        observed_masks = ~np.isnan(series)
        holidays = np.unique(np.where(~observed_masks)[0])
        gt_days = []
        for day in holidays:
            gt_days.append(day)
            # if day == 1:
            #     gt_days.append(day+1)
            # elif day == series.shape[0] - 1:
            #     gt_days.append(day - 1)
            # else:
            #     gt_days.append(day - 1)
            #     gt_days.append(day + 1)
        # for batch that there is no holiday, we add one day as a random holiday
        batch_splits = np.arange(0, B, batch_size)
        gt_days = np.unique(np.array(gt_days))
        gt_batch_inds = np.unique(np.digitize(gt_days, batch_splits)) - 1
        mask = ~np.isin(np.arange(0,batch_splits.shape[0]), gt_batch_inds)
        not_gt_batch_inds = np.arange(0, batch_splits.shape[0])[mask]
        for ind in not_gt_batch_inds:
            random_day = np.random.choice(np.arange(batch_splits[ind], batch_splits[ind+1]), size=int(np.ceil(batch_size/16)), replace=False)
            gt_days = np.append(gt_days, random_day)
        for ind in gt_batch_inds[:-1]:
            random_day = np.random.choice(np.arange(batch_splits[ind], batch_splits[ind + 1]))
            gt_days = np.append(gt_days, random_day)
        gt_days = np.unique(gt_days)
        gt_masks = observed_masks
        gt_masks[gt_days] = np.zeros_like(observed_masks[0], dtype=bool)
        series = np.nan_to_num(series)
        observed_values_tensor = tf.convert_to_tensor(series.astype('float32'))
        observed_masks_tensor = tf.convert_to_tensor(observed_masks.astype('float32'))
        gt_mask_tensor = tf.convert_to_tensor(gt_masks.astype('float32'))
        return [observed_values_tensor, observed_masks_tensor, gt_mask_tensor]

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
        # elif masking == 'holiday':
        #     sample = sample.numpy()


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
    return tfm.nlp.models.TransformerEncoder(num_layers=layers,
                                            dropout_rate=0.1,
                                            norm_first=False,
                                            norm_epsilon=1e-5,
                                            num_attention_heads=heads,
                                            intermediate_size=64,
                                            activation='gelu',) # (batch_size, input_length, hidden_size)



def Conv1d_with_init(in_channels, out_channels, kernel_size, initializer=None, activation=None, name=None):
    if initializer is None:
        initializer = tf.keras.initializers.HeNormal()
    layer = keras.layers.Conv1D(out_channels, kernel_size, data_format="channels_first", kernel_initializer=initializer,
                                activation=activation, name=name)
    return layer


def swish(x):
    return x * keras.activations.sigmoid(x)


class DiffusionEmbedding(keras.layers.Layer):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        setattr(self, "embedding", self._build_embedding(num_steps, int(embedding_dim / 2)))
        self.projection = keras.Sequential()
        self.projection.add(keras.layers.Dense(projection_dim, activation=swish))
        self.projection.add(keras.layers.Dense(projection_dim, activation=swish))

    @tf.function
    def call(self, t, training=True):
        x = tf.gather(self.embedding, t)
        x = self.projection(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = tf.cast(tf.expand_dims(tf.range(num_steps), 1), tf.float32)  # (T,1)
        frequencies = tf.cast(10.0 ** tf.expand_dims(tf.range(dim) / (dim - 1) * 4.0, 0), tf.float32)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
        table = tf.Variable(table, name="embedding", trainable=False)
        return table


class diff_CSDI(keras.Model): #layers.Layer
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.algo = config['time_layer']
        # feature first
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1, activation='relu', name='input_projection')

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1, activation='relu', name='output_projection1')
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1, tf.keras.initializers.Zeros(), name='output_projection2')

        # feature last
        self.residual_layers = []
        for _ in range(config["layers"]):
            self.residual_layers.append(
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    time_layer=config['time_layer'],
                    lmax=config['lmax'],
                )
            )

    def built_after_run(self):
        self.diffusion_embedding.built = True
        for i in range(len(self.residual_layers)):
            self.residual_layers[i].built = True
            if self.algo == 'S4':
                self.residual_layers[i].time_layer.built_after_run()

    # @tf.function(input_signature=[(tf.TensorSpec([None, 2, 6, 29], tf.float32),
    #                                tf.TensorSpec([None, 145, 6, 29], tf.float32),
    #                                tf.TensorSpec([None], tf.int32))])
    @tf.function
    def call(self, batch):
        x, cond_info, t = batch
        B, inputdim, K, L = x.shape
        x = rearrange(x, 'b d k l -> b d (k l)') # 64 2 14 100 -> 64 2 1400

        x = self.input_projection(x)
        x = rearrange(x, 'b c (k l) -> b c k l', k=K) # B channels K L
        diffusion_emb = self.diffusion_embedding(t)

        skip = tf.TensorArray(dtype=tf.float32, size=len(self.residual_layers))
        for i, layer in enumerate(self.residual_layers):
            x, skip_connection = layer((x, cond_info, diffusion_emb))
            skip = skip.write(i, skip_connection) #.mark_used() #

        x = tf.reduce_sum(skip.stack(), axis=0) / tf.math.sqrt(float(len(self.residual_layers)))
        # x = tf.reshape(x, [B, self.channels, K * L])
        x = rearrange(x, '... k l -> ... (k l)')
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = self.output_projection2(x)  # (B,1,K*L)
        # x = tf.reshape(x, [B, K, L])
        x = rearrange(tf.squeeze(x, 1), '... (k l) -> ... k l', k=K)
        return x


class ResidualBlock(keras.layers.Layer):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, time_layer='transformer', lmax=None):
        super().__init__()
        self.time_layer_type=time_layer
        self.diffusion_projection = keras.layers.Dense(channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)


        if time_layer=='transformer':
            self.time_layer = keras.Sequential()
            self.time_layer.add(keras.layers.Input((None, channels)))
            self.time_layer.add(tfm.nlp.models.TransformerEncoder(num_layers=1,
                                                                  dropout_rate=0.1,
                                                                  norm_first=False,
                                                                  norm_epsilon=1e-5,
                                                                  num_attention_heads=nheads,
                                                                  intermediate_size=64,
                                                                  activation='gelu',
                                                                  use_bias = True)) # (batch_size, input_length, hidden_size)
        elif time_layer=='S4':
            self.time_layer = S4Layer(features=channels, lmax=lmax)
        elif time_layer == 'S5':
            self.time_layer = S5Layer(ssm_size=16, features=channels) # ssm_size has Order(H)
        elif time_layer == 'Mega':
            self.time_layer = MegaLayer(features=channels, chunk_size=-1)
        self.feature_layer = keras.Sequential()
        self.feature_layer.add(keras.layers.Input((None, channels)))
        self.feature_layer.add(tfm.nlp.models.TransformerEncoder(num_layers=1,
                                                                  dropout_rate=0.1,
                                                                  norm_first=False,
                                                                  norm_epsilon=1e-5,
                                                                  num_attention_heads=nheads,
                                                                  intermediate_size=64,
                                                                  activation='gelu',
                                                                  use_bias = True)) # (batch_size, input_length, hidden_size)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = rearrange(y, '... c (k l) -> ... k c l', k=K) # b k c l
        # if self.time_layer_type == 'transformer':
        y = rearrange(y, ' b k c l ->  (b k) l c') # in torch version, batch_first is False so it transposes input as L B C but we dont need to do here
        y = self.time_layer(y) # output is bk l c
        y = rearrange(y, '(b k) l c -> b (k l) c', k=K) # transpose to b k l c
        y = rearrange(y, 'b (k l) c -> b c (k l)', k=K)  # b c (k l)
        # elif self.time_layer_type == 'S4' or self.time_layer_type == 'S5' or self.time_layer_type == 'Mega':
        #     y = rearrange(y, ' b k c l -> (b k) c l') # batch feature length
        #     y = self.time_layer(y)  # bk, c, l -> bk, l, c
        #     y = rearrange(y, '(b k) l c -> b c (k l)', k=K)  # b c k l

        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape

        if K == 1:
            return y
        y = rearrange(y, 'b c (k l) ->  (b l) c k', k=K) # (bl) c k
        y = rearrange(y, '(b l) c k -> (b l) k c', l=L)
        y = self.feature_layer(y) # (b l) k c
        y = rearrange(y, ' (b l) k c ->  b l k c', l=L) # b l k c
        y = rearrange(y, ' b l k c -> b c (k l)', l=L)
        return y

    @tf.function
    def call(self, batch):
        x, cond_info, diffusion_emb = batch
        B, channel, K, L = x.shape
        _, cond_dim, _, _ = cond_info.shape
        base_shape = x.shape
        x = rearrange(x, '... k l -> ... (k l)')

        diffusion_emb = self.diffusion_projection(diffusion_emb)
        diffusion_emb = tf.expand_dims(diffusion_emb, -1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,channel,K*L) -> (B,2*channel,K*L)

        cond_info = rearrange(cond_info, '... k l -> ... (k l)')
        cond_info = self.cond_projection(cond_info)  # (B, 2*channel,K*L)
        y = y + cond_info

        gate, filter = tf.split(y, 2, axis=1)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = tf.split(y, 2, axis=1)
        x = rearrange(x, '... (k l) -> ... k l ', k=K)
        residual = rearrange(residual, '... (k l) -> ... k l ', k=K)
        skip = rearrange(skip, '... (k l) -> ... k l ', k=K)
        return (x + residual) / math.sqrt(2.0), skip

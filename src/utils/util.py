import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from prettytable import PrettyTable
import keras.backend as K
from tqdm import tqdm

class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    # @tf.function
    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ["B", 'C', 'D', 'Lambda_re', 'Lambda_im', 'Delta']:

            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, weight / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_trainable_params = 0
    # total_non_trainable_params = 0
    for parameter in model.trainable_variables:
        name = parameter.name
        params = 1
        for k in parameter.shape:
            params *= k
        table.add_row([name, params])
        total_trainable_params += params

    print(table)
    print(f"Total Trainable Params: {total_trainable_params}")
    # print(f"Total non-Trainable Params: {total_non_trainable_params}")
    return total_trainable_params



def tensor_assign(input_tensor: tf.Tensor, positions: list, values: tf.float32) -> tf.Tensor:
    input_tensor = input_tensor.numpy()
    input_tensor[tuple(positions)] = values
    return input_tensor

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path

    Parameters:
    path (str): checkpoint path

    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        epoch = max(epoch, int(f))

    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, keras.Model):
        module_parameters = filter(lambda p: p.trainable, net.variables)
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """
    return tf.random.normal(size)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = tf.exp(tf.range(half_dim, dtype=tf.float32) * tf.constant(-_embed, dtype=tf.float32))
    _embed = tf.cast(diffusion_steps, _embed.dtype)* _embed
    diffusion_step_embed = tf.concat((tf.math.sin(_embed),tf.math.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """
    Beta = np.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    # Alpha_bar, Beta_tilde = tf.py_function(alpha_beta_bar_assign, inp=[Alpha, Beta, T], Tout=Alpha.dtype)
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Beta = tf.convert_to_tensor(Beta)
    Alpha = tf.convert_to_tensor(Alpha)
    Alpha_bar = tf.convert_to_tensor(Alpha_bar)
    Beta_tilde = tf.convert_to_tensor(Beta_tilde)
    Sigma = tf.math.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, tf.cast(Beta, tf.float32), tf.cast(Alpha, tf.float32), tf.cast(Alpha_bar, tf.float32), tf.cast(Sigma, tf.float32)
    diffusion_hyperparams = _dh
    return diffusion_hyperparams

# @tf.function
def sampling(net, diffusion_hyperparams, only_generate_missing, cond, mask, num_samples):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """
    size = cond.shape
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]

    current_sample_generator = lambda i: imputer(net,
                                                 T, Alpha, Alpha_bar, Sigma, size,
                                                 only_generate_missing, cond,
                                                 mask)
    # imputed_samples = tf.map_fn(current_sample_generator,
    #                             elems=[tf.range(num_samples)],
    #                             fn_output_signature=tf.TensorSpec(dtype=tf.float32, shape=size),
    #                             parallel_iterations=num_samples,
    #                             )
    # return imputed_samples
    pbar = tqdm(total=num_samples)
    # imputed_samples = tf.TensorArray(dtype=tf.float32, size=num_samples)
    imputed_samples = []
    sample_i = 0
    while sample_i < num_samples:
        current_sample = current_sample_generator(sample_i)
        imputed_samples.append(current_sample) # = imputed_samples.write(sample_i, current_sample)
        sample_i += 1
        if sample_i % 2 == 0 and sample_i > 0:
            pbar.update(2)
    return tf.stack(imputed_samples)


# @tf.function
def imputer(net, T, Alpha, Alpha_bar, Sigma, size, only_generate_missing, cond, mask):
    # pbar = tqdm(total=T)
    t = T - 1
    loss_mask = tf.constant(1.0) - mask
    # current_sample = tf.TensorArray(dtype=tf.float32, size=1, clear_after_read=False)
    # current_sample = current_sample.write(0, tf.random.normal(size, dtype=cond.dtype))
    current_sample = tf.random.normal(size, dtype=cond.dtype)
    while t >= 0:
        # if only_generate_missing == 1:
        current_sample = current_sample * (1.0 - mask) + cond * mask
        diffusion_steps = tf.cast(t * tf.ones((size[0], 1)), tf.int32)  # use the corresponding reverse step
        epsilon_theta = tf.stop_gradient(net(input_data=(current_sample, cond, mask),
                                             training=False))  # predict \epsilon according to \epsilon_\theta
        # update x_{t-1} to \mu_\theta(x_t)
        current_sample = (current_sample - (1 - Alpha[t]) / tf.math.sqrt(
            1 - Alpha_bar[t]) * epsilon_theta) / tf.math.sqrt(Alpha[t])
        if t > 0:
            current_sample = current_sample + Sigma[t] * std_normal(size)
        # add the variance term to x_{t-1}

        t -= 1
        # pbar.update(1)
    return current_sample


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """
    # net = tf.function(net)
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C is the dimension of each audio, L is audio length

    diffusion_steps = tf.random.uniform(shape=(B,), maxval=T, dtype=tf.int32) # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask + z * (1. - mask)
    transformed_X = tf.cast(tf.math.sqrt(tf.reshape(tf.gather(Alpha_bar,diffusion_steps), shape=[B, 1, 1])), dtype=audio.dtype)* audio + tf.cast(tf.math.sqrt(
        1 - tf.reshape(tf.gather(Alpha_bar,diffusion_steps), shape=[B, 1, 1])),dtype=z.dtype)* z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (transformed_X, cond, mask, tf.reshape(diffusion_steps, shape=(B, 1)),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)


def get_mask_rm(sample, k=None, rate=None):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    assert k is not None or rate is not None
    mask = np.ones(sample.shape)
    # mask = tf.Variable(mask_array, trainable=False)
    length_index = np.arange(mask.shape[0]) # lenght of series indexes
    for channel in range(mask.shape[1]):
        # perm = torch.randperm(len(length_index))
        perm = np.random.permutation(len(length_index))
        if rate is None:
            idx = perm[0:k]
        else:
            sample_num = int(mask.shape[0]*rate)
            idx = perm[0:sample_num]
        mask[:, channel][idx] = 0

    return tf.convert_to_tensor(mask)


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = np.ones(sample.shape)
    length_index = np.arange(mask.shape[0])
    list_of_segments_index = np.array_split(length_index, length_index.shape[0]//k + 1)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return tf.convert_to_tensor(mask)


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = np.ones(sample.shape)
    length_index = np.arange(mask.shape[0])
    list_of_segments_index = np.array_split(length_index, length_index.shape[0]//k + 1)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return tf.convert_to_tensor(mask)


def get_mask_holiday(sample):
    L, N, C = sample.shape
    sample = sample.transpose([1, 0, 2])
    gt_masks = []
    for s in sample:
        observed_masks = ~np.isnan(s)  # NA: 0 ;has data: 1
        holidays = np.unique(np.where(~observed_masks)[0])
        missing_ratio = len(holidays)/L
        gt_days = np.random.choice(np.unique(np.where(observed_masks)[0]), size=int(missing_ratio*L), replace=False)
        # B, C, L = sample.shape
        # observed_masks = ~np.isnan(sample) # NA: 0 ;has data: 1
        # holidays = np.unique(np.where(~observed_masks)[0])
        # gt_days = holidays
        # # for batch that there is no holiday, we add one day as a random holiday
        # batch_splits = np.arange(0, B, batch_size)
        # gt_batch_inds = np.unique(np.digitize(gt_days, batch_splits)) - 1
        # mask = ~np.isin(np.arange(0, batch_splits.shape[0]), gt_batch_inds)
        # not_gt_batch_inds = np.arange(0, batch_splits.shape[0])[mask]
        # for ind in not_gt_batch_inds:
        #     if ind < batch_splits.shape[0]-1:
        #         random_day = np.random.choice(np.arange(batch_splits[ind], batch_splits[ind + 1]),
        #                                       size=int(np.ceil(batch_size / 16)), replace=False)
        #         gt_days = np.append(gt_days, random_day)
        # for ind in gt_batch_inds:
        #     if ind < batch_splits.shape[0] - 1:
        #         random_day = np.random.choice(np.arange(batch_splits[ind], batch_splits[ind + 1]))
        #         gt_days = np.append(gt_days, random_day)
        gt_mask = observed_masks
        gt_mask[gt_days] = np.zeros_like(s[0], dtype=bool)
        gt_masks.append(gt_mask)
    gt_masks = np.array(gt_masks)

    return tf.cast(tf.convert_to_tensor(gt_masks), tf.float32) # N L C


import math
from functools import partial
import tensorflow as tf
from tensorflow import keras
from einops import rearrange
from scipy.fftpack import next_fast_len

# functions

def mask_fill(matrix, mask, num):
    return (matrix + (-(((mask * num) + num) - num)))

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    return val if exists(val) else d

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return tf.reshape(x, (*x.shape, *((1,) * num_dims)))

def conv1d_fft(x, weights, dim = -3, weight_dim = -2):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)
    x = tf.transpose(x) # B L K -> K L B
    f_x = tf.signal.rfft(x, fast_len)
    f_x = tf.transpose(f_x) # K L B -> B L K

    weights = rearrange(weights, '... j i -> ... i j')
    f_weight = tf.signal.rfft(weights, fast_len)
    f_weight = rearrange(f_weight, '... j i -> ... i j')

    f_v_weight = f_x * append_dims(tf.math.conj(f_weight), weight_dim - dim)
    out = tf.signal.irfft(tf.transpose(f_v_weight), fast_len)
    out = tf.transpose(out)
    out = tf.roll(out, -1, axis=(dim,))

    indices = tf.range(start=fast_len - N, limit=fast_len, dtype=tf.int32)
    out = tf.gather(out, indices=indices, axis=dim)
    return out

# positional bias for single-headed attention

class T5RelativePositionBias(keras.layers.Layer):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = keras.layers.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = tf.math.abs(n)
        else:
            n = tf.math.reduce_max(n, tf.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            tf.math.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = tf.math.reduce_min(val_if_large, tf.fill(tf.shape(val_if_large), num_buckets - 1))

        ret += tf.where(is_small, n, val_if_large)
        return ret

    def call(self, x):
        i, j = x.shape[-2:] # TODO *
        q_pos = tf.range(i, dtype = tf.int32)
        k_pos = tf.range(j, dtype=tf.int32)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# classes

class LaplacianAttnFn(keras.layers.Layer):
    def call(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt(0.25 * math.pi)
        return (1 + tf.math.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

class OffsetScale(keras.layers.Layer):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = tf.Variable(tf.ones(heads, dim), name='gamma')
        self.beta =  tf.Variable(tf.zeros(heads, dim), name='beta')
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev= 0.02)
        self.gamma.assign(initializer(shape=[heads, dim]))

    def call(self, x):
        out = tf.einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return tf.unstack(out, axis=-2)


class SingleHeadedAttention(keras.layers.Layer):
    def __init__(
        self,
        *,
        dim,
        dim_qk,
        dim_value,
        causal = False,
        laplacian_attn_fn = False
    ):
        super().__init__()
        self.causal = causal
        self.laplacian_attn_fn = laplacian_attn_fn

        self.attn_fn = partial(tf.nn.softmax, axis = -1) if not laplacian_attn_fn else LaplacianAttnFn()

        self.rel_pos_bias = T5RelativePositionBias(causal = causal, scale = dim_qk ** 0.5)

        self.to_qk = keras.Sequential([
            keras.layers.Input(shape=(dim)),
            keras.layers.Dense(dim_qk),
            keras.activations.swish()]
        )

        self.offsetscale = OffsetScale(dim_qk, heads = 2)

        self.to_v = keras.Sequential([
            keras.layers.Input(shape=(dim)),
            keras.layers.Dense(dim_value),
            keras.activations.swish()]
        )

    def call(self, x, v_input = None):
        seq_len, dim, dtype = *x.shape[-2:], x.dtype

        v_input = default(v_input, x)

        qk, v = self.to_qk(x), self.to_v(v_input)
        q, k = self.offsetscale(qk)

        scale = (seq_len ** -1) if self.laplacian_attn_fn else (dim ** -0.5)

        sim = tf.einsum('b i d, b j d -> b i j', q, k) * scale

        sim = sim + self.rel_pos_bias(sim)

        if self.causal:
            causal_mask = tf.ones((seq_len, seq_len), dtype = tf.bool) #.triu(1)
            causal_mask = tf.linalg.band_part(causal_mask, 0, -1) - tf.linalg.band_part(causal_mask, 0, 0)# upper triangle part

        if self.causal and not self.laplacian_attn_fn:
            # is softmax attention and using large negative value pre-softmax
            # sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
            sim = tf.where(causal_mask, -tf.reduce_max(tf.experimental.numpy.finfo(sim.dtype)), sim)

        attn = self.attn_fn(sim)

        if self.causal and self.laplacian_attn_fn:
            # if using laplacian attention function, zero out upper triangular with 0s
            # attn = attn.masked_fill(causal_mask, 0.)
            attn = tf.where(causal_mask, 0., attn)

        return tf.einsum('b i j, b j d -> b i d', attn, v)

class MultiHeadedEMA(keras.layers.Layer):
    def __init__(
        self,
        *,
        dim,
        heads,
        bidirectional = False,
        dim_head = None
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.expansion = tf.Variable(tf.random.normal(shape=[heads * (2 if bidirectional else 1), dim], dtype=tf.float32), name='expansion')
        self.reduction = tf.Variable(tf.random.normal(shape=[heads * (2 if bidirectional else 1), dim], dtype=tf.float32), name='reduction')

        # learned alpha and dampening factors

        self.alphas = tf.Variable(tf.random.normal(heads, dtype=tf.float32), name='alphas')
        self.dampen_factors = tf.Variable(tf.random.normal(heads, dtype=tf.float32), name='dampen_factors')

        if bidirectional:
            self.reverse_alphas = tf.Variable(tf.random.normal(heads, dtype=tf.float32), name='reverse_alphas')
            self.reverse_dampen_factors = tf.Variable(tf.random.normal(heads, dtype=tf.float32), name='reverse_dampen_factors')

    def call(self, x):
        seq_len = x.shape[1]

        # project in and split heads

        x = tf.einsum('... d, h d -> ... h d', x, self.expansion)

        if self.bidirectional:
            # x, x_reversed = x.chunk(2, dim = -2)
            x, x_reversed = tf.split(x, 2, axis=-2)
            x_reversed = tf.reverse(x_reversed, axis=[1])

        # weights derived from alphas (learned exponential smoothing decay rate)

        def apply_learned_ema_with_damping(x, alphas, dampen_factors):
            alphas = tf.nn.sigmoid(alphas)
            dampen_factors = tf.nn.sigmoid(dampen_factors)

            reversed_powers = tf.range(seq_len - 1, -1, -1)
            K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))

            # conv1d fft O(nlog(n))

            return conv1d_fft(x, K, dim = -3, weight_dim = -2)

        x = apply_learned_ema_with_damping(x, self.alphas, self.dampen_factors)

        if self.bidirectional:
            x_reversed = apply_learned_ema_with_damping(x_reversed, self.reverse_alphas, self.reverse_dampen_factors)
            x_reversed = tf.reverse(x_reversed, axis = [1])
            x = tf.concat((x, x_reversed), axis = -2)

        # combine heads and out

        return tf.einsum('... h d, h d -> ... d', x, self.reduction)

# Mega Layer
# Single headed Attention + Multi-headed EMA, then GRU-esque gating

class MegaLayer(keras.Model):
    def __init__(
        self,
        *,
        dim = 128,
        ema_heads = 16,
        attn_dim_qk = 64,
        attn_dim_value = 256,
        laplacian_attn_fn = False,
        causal = True,
        ema_dim_head = None
    ):
        super().__init__()

        self.single_headed_attn = SingleHeadedAttention(
            dim = dim,
            dim_qk = attn_dim_qk,
            dim_value = attn_dim_value,
            causal = causal,
            laplacian_attn_fn = laplacian_attn_fn
        )

        self.multi_headed_ema = MultiHeadedEMA(
            dim = dim,
            heads = ema_heads,
            bidirectional = not causal,
            dim_head = ema_dim_head
        )

        self.to_reset_gate = keras.Sequential([
            keras.layers.Input(shape=(dim,)),
            keras.layers.Dense(attn_dim_value),
            keras.activations.swish()]
        )

        self.to_update_gate = keras.Sequential([
            keras.layers.Input(shape=(dim,)),
            keras.layers.Dense(dim),
            keras.activations.sigmoid()]
        )

        # equation 14, for calculating H

        self.Wh = tf.Variable(tf.random.normal(dim, dim), name='Wh')
        self.Uh = tf.Variable(tf.random.normal(attn_dim_value, dim), name='Uh')
        self.bh = tf.Variable(tf.random.normal(dim), name='bh')

    def call(self, x, residual = None):
        residual = default(residual, x)

        ema_output = self.multi_headed_ema(x)
        attn_output = self.single_headed_attn(ema_output, x)

        reset_gate = self.to_reset_gate(ema_output)
        update_gate = self.to_update_gate(ema_output)

        gated_attn_output = attn_output * reset_gate

        # equation 14

        H = keras.activations.swish(ema_output @ self.Wh + gated_attn_output @ self.Uh + self.bh)

        # update gate

        return update_gate * H + (1 - update_gate) * residual

# Mega

def FeedForward(dim, ff_mult):
    dim_hidden = int(dim * ff_mult)
    return keras.Sequential([
        keras.layers.Input(shape=(dim,)),
        keras.layers.Dense(dim_hidden),
        keras.activations.gelu(),
        keras.layers.Dense(dim)
    ])


class Mega(keras.Model):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        ff_mult = 2,
        pre_norm = False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = keras.layers.Embedding(num_tokens, dim)
        self.pre_norm = pre_norm

        self.layers = []
        for _ in range(depth):
            self.layers.append(keras.Sequential([
                MegaLayer(dim = dim, **kwargs),
                keras.layers.LayerNormalization(dim),
                FeedForward(dim = dim, ff_mult = ff_mult),
                keras.layers.LayerNormalization(dim),
            ]))

        # self.to_logits = keras.Sequential([
        #     keras.layers.LayerNormalization(dim) if pre_norm else tf.identity(),
        #     keras.layers.Dense(num_tokens)
        # ])

    def forward(self, x):
        pre_norm = self.pre_norm
        post_norm = not self.pre_norm

        x = self.token_emb(x)

        for mega_layer, mega_norm, ff, ff_norm in self.layers:
            mega_maybe_prenorm = mega_norm if pre_norm else identity
            ff_maybe_prenorm = ff_norm if pre_norm else identity

            mega_maybe_postnorm = mega_norm if post_norm else identity
            ff_maybe_postnorm = ff_norm if post_norm else identity

            x = mega_layer(mega_maybe_prenorm(x), x)

            x = mega_maybe_postnorm(x)

            x = ff(ff_maybe_prenorm(x)) + x

            x = ff_maybe_postnorm(x)

        return x

import math
from functools import partial
import tensorflow as tf
from tensorflow import keras
from einops import rearrange
from scipy.fftpack import next_fast_len

# functions

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

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = tf.signal.rfft(x, fast_len ) #, dim = dim) # TODO dim
    f_weight = tf.signal.rfft(weights, fast_len) #, dim = weight_dim) # TODO dim

    f_v_weight = f_x * append_dims(tf.math.conj(f_weight), weight_dim - dim)
    out = tf.signal.irfft(f_v_weight, fast_len) #, dim = dim) # TODO dim
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

    def forward(self, x):
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
    def forward(self, x):
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

    def forward(self, x):
        out = tf.einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return tf.unstack(out, axis=-2)

# TODO tonight is here
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

        self.attn_fn = partial(F.softmax, dim = -1) if not laplacian_attn_fn else LaplacianAttnFn()

        self.rel_pos_bias = T5RelativePositionBias(causal = causal, scale = dim_qk ** 0.5)

        self.to_qk = nn.Sequential(
            nn.Linear(dim, dim_qk),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(dim_qk, heads = 2)

        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_value),
            nn.SiLU()
        )

    def forward(self, x, v_input = None):
        seq_len, dim, device, dtype = *x.shape[-2:], x.device, x.dtype

        v_input = default(v_input, x)

        qk, v = self.to_qk(x), self.to_v(v_input)
        q, k = self.offsetscale(qk)

        scale = (seq_len ** -1) if self.laplacian_attn_fn else (dim ** -0.5)

        sim = einsum('b i d, b j d -> b i j', q, k) * scale

        sim = sim + self.rel_pos_bias(sim)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)

        if self.causal and not self.laplacian_attn_fn:
            # is softmax attention and using large negative value pre-softmax
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = self.attn_fn(sim)

        if self.causal and self.laplacian_attn_fn:
            # if using laplacian attention function, zero out upper triangular with 0s
            attn = attn.masked_fill(causal_mask, 0.)

        return einsum('b i j, b j d -> b i d', attn, v)

class MultiHeadedEMA(nn.Module):
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

        self.expansion = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))
        self.reduction = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))

        # learned alpha and dampening factors

        self.alphas = nn.Parameter(torch.randn(heads))
        self.dampen_factors = nn.Parameter(torch.randn(heads))

        if bidirectional:
            self.reverse_alphas = nn.Parameter(torch.randn(heads))
            self.reverse_dampen_factors = nn.Parameter(torch.randn(heads))

    def forward(self, x):
        device, seq_len = x.device, x.shape[1]

        # project in and split heads

        x = einsum('... d, h d -> ... h d', x, self.expansion)

        if self.bidirectional:
            x, x_reversed = x.chunk(2, dim = -2)
            x_reversed = torch.flip(x_reversed, dims = (1,))

        # weights derived from alphas (learned exponential smoothing decay rate)

        def apply_learned_ema_with_damping(x, alphas, dampen_factors):
            alphas = alphas.sigmoid()
            dampen_factors = dampen_factors.sigmoid()

            reversed_powers = torch.arange(seq_len - 1, -1, -1, device = device)
            K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))

            # conv1d fft O(nlog(n))

            return conv1d_fft(x, K, dim = -3, weight_dim = -2)

        x = apply_learned_ema_with_damping(x, self.alphas, self.dampen_factors)

        if self.bidirectional:
            x_reversed = apply_learned_ema_with_damping(x_reversed, self.reverse_alphas, self.reverse_dampen_factors)
            x_reversed = torch.flip(x_reversed, dims = (1,))
            x = torch.cat((x, x_reversed), dim = -2)

        # combine heads and out

        return einsum('... h d, h d -> ... d', x, self.reduction)

# Mega Layer
# Single headed Attention + Multi-headed EMA, then GRU-esque gating

class MegaLayer(nn.Module):
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

        self.to_reset_gate = nn.Sequential(
            nn.Linear(dim, attn_dim_value),
            nn.SiLU()
        )

        self.to_update_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # equation 14, for calculating H

        self.Wh = nn.Parameter(torch.randn(dim, dim))
        self.Uh = nn.Parameter(torch.randn(attn_dim_value, dim))
        self.bh = nn.Parameter(torch.randn(dim))

    def forward(self, x, residual = None):
        residual = default(residual, x)

        ema_output = self.multi_headed_ema(x)
        attn_output = self.single_headed_attn(ema_output, x)

        reset_gate = self.to_reset_gate(ema_output)
        update_gate = self.to_update_gate(ema_output)

        gated_attn_output = attn_output * reset_gate

        # equation 14

        H = F.silu(ema_output @ self.Wh + gated_attn_output @ self.Uh + self.bh)

        # update gate

        return update_gate * H + (1 - update_gate) * residual

# Mega

def FeedForward(dim, ff_mult):
    dim_hidden = int(dim * ff_mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Linear(dim_hidden, dim)
    )

class Mega(nn.Module):
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
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pre_norm = pre_norm

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MegaLayer(dim = dim, **kwargs),
                nn.LayerNorm(dim),
                FeedForward(dim = dim, ff_mult = ff_mult),
                nn.LayerNorm(dim)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim) if pre_norm else nn.Identity(),
            nn.Linear(dim, num_tokens)
        )

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

        return self.to_logits(x)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
from functools import partial
from scipy import special as ss
from einops import rearrange, repeat
import opt_einsum as oe

# rearrange = tf.function(rearrange)
contract = oe.contract
contract_expression = oe.contract_expression

# _c2r = torch.view_as_real
# _r2c = torch.view_as_complex
_conj = lambda x: tf.concat([x, tf.math.conj(x)], axis=-1)
# if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
#     _resolve_conj = lambda x: x.conj().resolve_conj()
# else:
_resolve_conj = lambda x: tf.math.conj(x)

def _c2r(c):
    return tf.stack([tf.math.real(c), tf.math.imag(c)], axis=-1)

def _r2c(r):
    return tf.squeeze(tf.complex(r[..., :-1], r[..., -1:]), -1)

""" simple keras.Model components """

def Activation(activation=None):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return tf.identity
    elif activation == 'tanh':
        return keras.activations.tanh
    elif activation == 'relu':
        return keras.activations.relu
    elif activation == 'gelu':
        return keras.activations.gelu
    elif activation in ['swish', 'silu']:
        return keras.activations.silu
    # TODO GLU activation layer https://medium.com/deeplearningmadeeasy/glu-gated-linear-unit-21e71cd52081
    # elif activation == 'glu':
    #     return keras.activations.glu(dim=dim) # Gated LU is not implemented yet but we can self implement this late
    elif activation == 'sigmoid':
        return keras.activations.sigmoid
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        # initializer = partial(keras.initializers.HeUniform, nonlinearity=nonlinearity)
        initializer = keras.initializers.HeUniform
    elif name == 'normal':
        initializer = keras.initializers.HeNormal
        # initializer = partial(keras.initializers.HeNormal, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = keras.initializers.GlorotNormal
    elif name == 'zero':
        initializer = partial(keras.initializers.Constant, value=0)
    elif name == 'one':
        initializer = partial(keras.initializers.Constant, value=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

class TransposedLinear(keras.Model):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, initializer = None, bias=True):
        super().__init__()

        # self.weight = tf.Variable(tf.zeros(d_output, d_input))
        if initializer is None:
            w_init =  keras.initializers.HeUniform()
        else:
            w_init = initializer
        self.w = tf.Variable(
            initial_value= w_init(shape=(d_input, d_output), dtype="float32"),
            trainable=True, name='transposed_weight'
        )
        # keras.initializers.HeUniform(self.weight, a=math.sqrt(5)) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            # self.bias = tf.Variable(tf.zeros(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            if initializer is None:
                b_init = keras.initializers.RandomUniform(-bound, bound)
            else:
                b_init = initializer
            self.b = tf.Variable(
                initial_value=b_init(shape=(d_output,1), dtype="float32"), trainable=True, name='transposed_bias'
            )

        else:
            self.b = 0.0

    def call(self, x):
        return  tf.einsum('... u l, v u -> ... v l', x, self.w) + self.b  #contract('... u l, v u -> ... v l', x, self.w) + self.b

def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear keras.Model with control over axes order, initialization, and activation """

    # Construct core module
    if activation == 'glu': d_output *= 2
    linear =  TransposedLinear(d_input, d_output, bias=bias, initializer=initializer) \
        if transposed else keras.layers.Dense(units=d_output, kernel_initializer=initializer, activation=activation) #linear_cls(d_input, d_output, bias=bias, initializer=initializer, **kwargs)

    # Initialize weight
    if initializer is not None:
        initializer = get_initializer(initializer)

    # Initialize bias
    if bias and zero_bias_init:
        if transposed:
            linear.b = keras.initializers.Zeros(shape=(d_output, ), value=0)
        else:
            linear.bias = keras.initializers.Zeros(shape=(d_output, ), value=0)

    # Weight norm
    # if weight_norm:
    #     linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation)
        linear.add(keras.layers.Activation(activation))
    return linear

# TODO logger
# def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
#     """Initializes multi-GPU-friendly python logger."""
#
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#
#     # this ensures all logging levels get marked with the rank zero decorator
#     # otherwise logs would get multiplied for each GPU process in multi-GPU setup
#     for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
#         setattr(logger, level, rank_zero_only(getattr(logger, level)))
#
#     return logger
# log = get_logger(__name__)
#

def cauchy_slow(v, z, w):
    """
    v, w: (..., N)
    z: (..., L)
    returns: (..., L)
    """
    cauchy_matrix = tf.expand_dims(v, -1) / (tf.expand_dims(z, -2) - tf.expand_dims(w, -1))  # (... N L)
    return tf.reduce_sum(cauchy_matrix, axis=-2)

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    # I = torch.eye(A.shape[-1]).to(A)
    I = tf.eye(A.shape[-1], dtype=A.dtype)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    # k = v.size(-1) - l
    k = v.shape[-1] - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    # while v.size(-1) > 1:
    while v.shape[-1] > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


""" HiPPO utilities """

def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=tf.float32):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = tf.sqrt(.5+tf.range(N, dtype=dtype)) # (1 N)
        P = tf.expand_dims(P, axis=0)
    elif measure == 'legt':
        assert rank >= 2
        P = tf.sqrt(1+2*tf.range(N, dtype=dtype)) # (N)
        P0 = tf.identity(P) #clone
        P0[0::2] = 0.
        P1 = tf.identity(P) #clone
        P1[1::2] = 0.
        P = tf.stack([P0, P1], aixs=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * tf.ones(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = tf.ones(N, dtype=dtype) # (N)
        P0 = tf.identity(P) #clone
        P0[0::2] = 0.
        P1 = tf.identity(P) #clone
        P1[1::2] = 0.
        P = tf.stack([P0, P1], axis=0) # (2 N)
    else: raise NotImplementedError

    d = P.shape[0]
    if rank > d:
        P = tf.concat([P, tf.zeros(rank-d, N, dtype=dtype)], axis=0) # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=tf.float32):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == tf.float32 or tf.complex64
    if measure == 'random':
        dtype = tf.complex64 if dtype == tf.float32 else tf.complex128
        # w = torch.randn(N//2, dtype=dtype)
        w = -tf.exp(tf.random.normal([N//2])) + 1j*tf.random.normal([N//2])
        P = tf.random.normal(shape = [rank, N//2], dtype=dtype)
        B = tf.random.normal([N//2], dtype=dtype)
        V = tf.eye(N, dtype=dtype)[..., :N//2] # Only used in testing
        return w, P, B, V

    A, B = transition(measure, N)
    A = tf.convert_to_tensor(A, dtype=dtype) # (N, N)
    B = tf.convert_to_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    # AP = A + tf.reduce_sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)
    AP = A + tf.reduce_sum(tf.expand_dims(P, axis =-2) * tf.expand_dims(P, axis=-1), axis=-3)
    w, V = tf.linalg.eig(AP) # (..., N) (..., N, N)

    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2] #.contiguous()
    V = V[..., 0::2] #.contiguous()

    # V_inv = tf.math.conj(V) #.transpose(-1, -2)
    V_inv = tf.transpose(V, perm = [1, 0], conjugate=True)

    # real_V_inv = tf.math.real(V_inv)
    # imag_V_inv = tf.math.imag(V_inv)
    # B_real = tf.einsum('ij, j -> i', real_V_inv, B) # V^* B
    # B_imag = tf.einsum('ij, j -> i', imag_V_inv, B)
    # B = tf.cast(tf.complex(B_real, B_imag), dtype=V_inv.dtype)
    # P_real = tf.einsum('ij, ...j -> i', real_V_inv, P) # V^* P
    # P_imag = tf.einsum('ij, ...j -> ...i', imag_V_inv, P)
    # P = tf.cast(tf.complex(P_real, P_imag), dtype=V_inv.dtype)

    B = tf.matmul(V, tf.expand_dims(tf.complex(B, tf.zeros_like(B)), 1), adjoint_a=True)
    B = tf.squeeze(B) # (N/2, )
    P = tf.matmul(tf.complex(P, tf.zeros_like(P)), tf.math.conj(V))

    return w, P, B, V


def bilinear(dt, A, B=None):
    """
    dt: (...) timescales
    A: (... N N)
    B: (... N)
    """
    N = A.shape[-1]
    I = tf.eye(N, dtype=A.dtype)
    A_backwards = I - dt[:, None, None] / 2 * A
    A_forwards = I + dt[:, None, None] / 2 * A

    if B is None:
        dB = None
    else:
        dB = dt[..., None] * tf.linalg.solve(
            A_backwards,tf.expand_dims(B, axis=-1)
        ) #.squeeze(-1)
        dB = tf.squeeze(dB, axis=-1) # (... N)

    dA = tf.linalg.solve(A_backwards, A_forwards)  # (... N N)
    return dA, dB


class SSKernelNPLR(keras.layers.Layer):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """

    # @torch.no_grad() in torch the gradient will automatically traced, but tensorflow needs tf.GradientTape()
    def _setup_C(self, double_length=False):
        """ Construct C~ from C

        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        self._setup_state()
        dA_L = power(self.L, self.dA)

        dA_L = tf.transpose(dA_L, perm = [0, 2, 1])
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = tf.einsum("h m n, c h n -> c h m", dA_L, C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C.assign(tf.identity(_c2r(C_)))

        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, cache=True)

    def _omega(self, L, dtype, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes """
        omega = np.exp(-2j * np.pi / (L))
        omega = omega ** np.arange(0, L // 2 + 1)
        omega = tf.constant(omega, dtype=dtype)  # \omega_{2L}
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            setattr(self, 'omega', tf.Variable(_c2r(omega), name='omega', trainable=False))
            setattr(self, 'z', tf.Variable(_c2r(z), name='z', trainable=False))
        return omega, z

    def __init__(
        self,
        L, w, P, B, C, log_dt,
        hurwitz=False,
        trainable=None,
        lr=None,
        tie_state=False,
        length_correction=True,
        verbose=False,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*

        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)

        hurwitz: tie pq and ensure w has negative real part
        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)
        tie_state: tie all state parameters across the H hidden features
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.verbose = verbose

        # Rank of low-rank correction
        self.rank = P.shape[-2]
        assert w.shape[-1] == P.shape[-1] == B.shape[-1] == C.shape[-1]
        self.H = log_dt.shape[-1]
        self.N = w.shape[-1]

        # Broadcast everything to correct shapes
        board_cast_shape = tf.broadcast_dynamic_shape(C.shape, (1, self.H, self.N))
        expanded_shape = tf.cast(board_cast_shape/C.shape, dtype=tf.int32)
        C = tf.tile(C,expanded_shape) # (H, C, N)
        H = 1 if self.tie_state else self.H
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, cache=True)

        # Register parameters
        # C is a regular parameter, not state
        # self.C = nn.Parameter(_c2r(C.conj().resolve_conj()))
        # self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        self.C = tf.Variable(_c2r(tf.math.conj(C)), name='C')
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True
        self.register("log_dt", log_dt, trainable.get('dt', train), lr, 0.0)
        self.register("B", _c2r(B), trainable.get('B', train), lr, 0.0)
        self.register("P", _c2r(P), trainable.get('P', train), lr, 0.0)
        if self.hurwitz:
            log_w_real = tf.math.log(-w.real + 1e-3) # Some of the HiPPO methods have real part 0
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.get('A', 0), lr, 0.0)
            self.register("w_imag", w_imag, trainable.get('A', train), lr, 0.0)
            self.Q = None
        else:
            self.register("w", _c2r(w), trainable.get('A', train), lr, 0.0)
            # self.register("Q", _c2r(P.clone().conj().resolve_conj()), trainable.get('P', train), lr, 0.0)
            Q = tf.math.conj(tf.identity(P))
            self.register("Q", _c2r(Q), trainable.get('P', train), lr, 0.0)

        if length_correction:
            self._setup_C()

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            w_real = -tf.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        return w

    def call(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        while rate * L > self.L:
            self.double_length()

        dt = tf.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = tf.math.conj(P) if self.Q is None else _r2c(self.Q)
        w = self._w()

        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
        else:
            omega, z = self._omega(int(self.L/rate), dtype=w.dtype, cache=False)

        if self.tie_state:
            B = repeat(B, '... 1 n -> ... h n', h=self.H)
            P = repeat(P, '... 1 n -> ... h n', h=self.H)
            Q = repeat(Q, '... 1 n -> ... h n', h=self.H)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.shape[-1] == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - tf.einsum('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / tf.expand_dims(dt, -1) + sA / 2
            s = s[..., :self.N]

            B = tf.concat([s, B], axis=-3)  # (s+1, H, N)

        # Incorporate dt into A

        real_w = tf.math.real(w) * tf.expand_dims(dt, -1)
        imag_w = tf.math.imag(w) * tf.expand_dims(dt, -1)
        w = tf.cast(tf.complex(real_w, imag_w), dtype=tf.complex64) # (H N)

        # Stack B and p, C and q for convenient batching
        B = tf.concat([B, P], axis=-3) # (s+1+r, H, N)
        C = tf.concat([C, Q], axis=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = tf.expand_dims(B, -3) * tf.expand_dims(C, -4)  # (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L)

        # TODO cauchy function only has slow version
        # Calculate resolvent at omega
        # if has_cauchy_extension and z.dtype == tf.complex64:
        #     r = cauchy_mult(v, z, w, symmetric=True)
        # elif has_pykeops:
        #     r = cauchy_conj(v, z, w)
        # else:
        r = cauchy_slow(v, z, w)
        # if r.dtype == dt.dtype:
        #     r = r * dt[None, None, :, None]  # (S+1+R, C+R, H, L)
        # else:
        r = _r2c(_c2r(r) * dt[None, None, :, None, None])

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = tf.linalg.inv(tf.eye(self.rank) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - tf.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = tf.signal.irfft(k_f)  # (S+1, C, H, L)

        # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)
        return k_B, k_state

    # @torch.no_grad()
    # TODO
    def double_length(self):
        if self.verbose: log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(double_length=True)

    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B) # (H N)
        P = _r2c(self.P)
        Q = tf.math.conj(P) if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = tf.exp(self.log_dt)
        real_D = 2.0 / tf.expand_dims(dt, -1) - tf.math.real(w)
        imag_D = tf.math.imag(w)
        D = tf.math.reciprocal(tf.complex(real_D, imag_D))  # (H, N)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        real_R = tf.eye(self.rank) + 2 * tf.math.real(tf.einsum('r h n, h n, s h n -> h r s', Q, D, P))
        imag_R = tf.zeros_like(real_R)
        R = tf.cast(tf.complex(real_R, imag_R) , dtype=Q_D.dtype) # (H r r)
        R = tf.linalg.solve(R, Q_D) # (H r N)
        R = rearrange(R, 'h r n -> r h n')
        real_E = 2.0 / tf.expand_dims(dt, -1) + tf.math.real(w)
        imag_E = tf.math.imag(w)
        E = tf.complex(real_E, imag_E)

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": E, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = tf.zeros(self.H, dtype=C.dtype)
        if state is None: # Special case used to find dB
            state = tf.zeros([self.H, self.N], dtype=C.dtype)

        step_params = self.step_params.copy()
        if state.shape[-1] == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: tf.einsum('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.shape[-1] == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: tf.einsum('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)
        # if state.shape[-1] == self.N:
        #     new_state = E * state - tf.einsum('r h n, r h m, ... h m -> ... h n', _conj(P), _conj(Q), _conj(state))  # (B H N)
        #     # new_state = E * state - oe.contract('r h n, r h m, ... h m -> ... h n', P, Q, state)
        #     new_state = new_state + 2.0 * B * tf.expand_dims(u, -1)  # (B H N)
        #     new_state = D * (new_state - tf.einsum('r h n, r h m, ... h m -> ... h n', _conj(P), _conj(R), _conj(new_state)))
        # else:
        if state.shape[-2] != P.shape[-2]:
            multiples = [1] * len(state.shape)
            multiples[-2] = P.shape[-2]//state.shape[-2]
            contrator_state = tf.tile(state, multiples)
        else:
            contrator_state = state
        new_state = E * state - contract_fn(P, Q, contrator_state) # (B H N)
        # new_state = E * state - oe.contract('r h n, r h m, ... h m -> ... h n', P, Q, state)
        new_state = new_state + 2.0 * B * tf.expand_dims(u, -1)  # (B H N)
        new_state = D * (new_state - contract_fn( P, R, new_state))

        return new_state

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = tf.expand_dims(tf.eye(2*self.N, dtype=C.dtype), -2) # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA # (H N N)

        u = tf.ones(self.H, dtype=C.dtype)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n') # (H N)

    def _step_state(self, u, state):
        """ Must be called after self.default_state() is used to construct an initial state!  """
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(self.dB, u)
        return next_state


    def setup_step(self, mode='dense'):
        """ Set up dA, dB, dC discretized parameters for stepping """
        self._setup_state()

        # Calculate original C
        dA_L = power(self.L, self.dA)
        I = tf.eye(self.dA.size(-1), dtype=dA_L.dtype)
        C = _conj(_r2c(self.C)) # (H C N)

        dC = tf.linalg.solve(
            I - tf.transpose(dA_L, perm = [-1, -2]),
            tf.expand_dims(C, -1),
        )
        dC = tf.squeeze(dC, -1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = tf.linalg.eigvals(self.dA)
            V_inv = tf.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            # TODO 2-norm of two tensors in tensorflow
            if self.verbose:
                print("Diagonalization error:", tf.norm(V @ tf.linalg.tensor_diag(L) @ V_inv - self.dA, ord='euclidean'))
                # print("Diagonalization error:", torch.dist(V @ tf.linalg.tensor_diag(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = tf.einsum('h n m, h m -> h n', V_inv, self.dB)
            self.dC = tf.einsum('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")


    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.shape[-1]
        H = C.shape[-2]

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode !='linear':
            N *= 2

            if self._step_mode == 'diagonal':
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N), # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N), # self.dC.shape
            batch_shape + (H, N),
        )
        state = tf.zeros(*batch_shape, H, N, dtype=C.dtype)
        return state

    def step(self, u, state):
        """ Must have called self.setup_step() and created state with self.default_state() before calling this """

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""
        setattr(self, name,  tf.Variable(tensor, name=name, trainable=trainable))
        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


class HippoSSKernel(keras.layers.Layer):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
            self,
            H,
            N=64,
            L=1,
            measure="legs",
            rank=1,
            channels=1,  # 1-dim to C-dim map; can think of C as having separate "heads"
            dt_min=0.001,
            dt_max=0.1,
            trainable=None,  # Dictionary of options to train various HiPPO parameters
            lr=None,  # Hook to set LR of hippo parameters differently
            length_correction=True,
            # Multiply by I-A|^L after initialization; can be turned off for initialization speed
            hurwitz=False,
            tie_state=False,  # Tie parameters of HiPPO ODE across the H features
            precision=1,  # 1 (single) or 2 (double) for the kernel
            resample=False,
            # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
            verbose=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = tf.double if self.precision == 2 else tf.float32
        cdtype = tf.complex64 if dtype == tf.float32 else tf.complex128
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        log_dt = tf.random.uniform([self.H], dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
        C = tf.complex(tf.random.normal([channels, self.H, self.N // 2], dtype=dtype), tf.random.normal([channels, self.H, self.N // 2], dtype=dtype))
        # C = tf.random.normal([channels, self.H, self.N // 2], dtype=cdtype)
        self.kernel = SSKernelNPLR(
            L, w, p, B, C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )

    def call(self, L=None):
        k, _ = self.kernel.call(rate=self.rate, L=L)
        return tf.cast(k, dtype=tf.float32)

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


# TODO not sure this is the same with torch encoder
def get_tf_trans(heads=8, layers=1, channels=64):
    # encoder_layer = nn.TransformerEncoderLayer(
    #     d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    encoder_inputs = keras.Input(shape=(channels,), name="encoder_inputs")
    outputs = encoder_inputs
    for _ in range(layers):
        outputs = keras.keras_nlp.TransformerEncoder(
            intermediate_dim=64, num_heads=heads, activation="gelu"
        )(input=outputs)

    return keras.Model(encoder_inputs, outputs, name="encoder")


class S4(keras.layers.Layer):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1,
            # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1,  # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu',  # activation in between SS and FF
            postact=None,  # activation after FF
            initializer=None,  # initializer on FF
            weight_norm=False,  # weight normalization on FF
            hyper_act=None,  # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True,  # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
    ):

        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        # if verbose:
        #     import src.utils.train
        #     log = src.utils.train.get_logger(__name__)
        #     log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = tf.Variable(tf.random.normal(shape=[channels, self.h]), name='D')

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose,
                                    **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = keras.layers.SpatialDropout2D if self.transposed else keras.layers.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else tf.identity

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h * self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

    @tf.function
    def call(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = tf.transpose(u, [0, 2, 1])
        L = u.shape[-1]

        # Compute SS Kernel
        k = self.kernel.call(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            # k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k0, k1 = k[0, None], k[1, None]
            k = tf.pad(k0, [[0,0],[0,0],[0,L]]) + tf.pad(tf.reverse(k1, [-1]), [[0,0],[0,0],[L, 0]])
        else:
            k = k
        k_f = tf.signal.rfft(k, [2 * L])  # (C H L)
        u_f = tf.signal.rfft(u, [2 * L])  # (B H L)
        y_f = tf.einsum('bhl,chl->bchl', u_f, k_f)  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = tf.signal.irfft(y_f, [2 * L])[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + tf.einsum('bhl,ch->bchl', u, self.D)  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(tf.transpose(self.activation(y), perm=[0, 2, 1])) # dropout needs input of (samples, timesteps, channels) which is B L C
        y = tf.transpose(y, perm=[0, 2, 1]) # B C L

        if not self.transposed: y = tf.transpose(y, perm=[0, 2, 1])

        y = self.output_linear(y)

        return y, None

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + tf.expand_dims(u, -2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(tf.expand_dims(y, -1))
            y = tf.squeeze(y, -1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class S4Layer(keras.Model):
    # S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
        super().__init__()
        self.s4_layer = S4(d_model=features,
                           d_state=N,
                           l_max=lmax,
                           bidirectional=bidirectional)

        self.norm_layer = keras.layers.LayerNormalization(axis=-1) if layer_norm else tf.identity
        self.dropout = keras.layers.SpatialDropout1D(dropout) if dropout > 0 else tf.identity

    @tf.function
    def call(self, x):
        # x has shape # batch, feature, seq
        # x = tf.transpose(x, perm=[1, 2, 0])  (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer.call(x)  # batch, feature,  seq,
        xout = self.dropout(tf.transpose(xout, perm=[0,2,1]))
        xout = xout + tf.transpose(x, perm=[0,2,1])  # skip connection   # batch, seq, feature
        return self.norm_layer(xout) # apply normalization to features

    def built_after_run(self):
        self.built=True
        self.s4_layer.built = True
        self.s4_layer.kernel.built = True
        self.s4_layer.kernel.kernel.built = True



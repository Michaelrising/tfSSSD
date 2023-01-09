import tensorflow as tf
from functools import partial
import tensorflow_probability as tfp
from tensorflow import keras

def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = tf.math.sqrt(1 + 2 * tf.range(N))
    A = tf.expand_dims(P, -1) * tf.expand_dims(P, 0)
    A = tf.linalg.LinearOperatorLowerTriangular(A) - tf.linalg.diag(tf.range(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = tf.math.sqrt(tf.range(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = tf.math.sqrt(2 * tf.range(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + tf.expand_dims(P, -1) * tf.expand_dims(P, 0)

    S_diag = tf.linalg.diag(S)
    Lambda_real = tf.reduce_mean(S_diag) * tf.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = tf.linalg.eigvals(S * -1j)

    P = tf.math.conj(V) @ P
    B_orig = B
    B = tf.math.conj(V) @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(key, shape):
        """ Init function
             Args:
                 key: random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return tf.random.uniform(shape, seed=key) * (
            tf.math.log(dt_max) - tf.math.log(dt_min)
        ) + tf.math.log(dt_min)

    return init

# TODO random key in jax
def init_log_steps(key, input):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = tf.random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)
    log_step = tf.stack(log_steps)

    return log_step


def init_VinvB(shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = tf.random.truncated_normal(shape)
    VinvB = Vinv @ B
    VinvB_real = tf.math.real(VinvB)
    VinvB_imag = tf.math.imag(VinvB)
    return tf.concat((tf.expand_dims(VinvB_real, -1), tf.expand_dims(VinvB_imag, -1)), axis=-1)


def trunc_standard_normal(key, shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = tf.random.split(key)
        C = tf.random.truncated_normal(shape=(1, P, 2), seed=skey)
        Cs.append(C)
    return tf.stack(Cs)[:, 0]


def init_CV(init_fun, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(shape=shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = tf.math.real(CV)
    CV_imag = tf.math.imag(CV)
    return tf.concat((tf.expand_dims(CV_real, -1), tf.expand_dims(CV_imag, -1)), axis=-1)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = tf.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    # Lambda_bar = tf.Variable(Lambda_bar, trainable=True, name='Lambda_bar')
    # B_bar = tf.Variable(B_bar, trainable=True, name='B_bar')
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = tf.ones(Lambda.shape[0])
    Lambda_bar = tf.math.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    # Lambda_bar = tf.Variable(Lambda_bar, trainable=True, name='Lambda_bar')
    # B_bar = tf.Variable(B_bar, trainable=True, name='B_bar')
    return Lambda_bar, B_bar


# Parallel scan operations
# @jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * tf.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = tf.vectorized_map(lambda u: B_bar @ u)(input_sequence) # TODO jax.vmap to tensorflow vectorized_map

    # TODO scan associate reverse
    # _, xs = tfp.math.scan_associative(fn=binary_operator, elems=[Lambda_elements, Bu_elements])
    _, xs = tf.scan(tf.vectorized_map(binary_operator), [Lambda_elements, Bu_elements])

    if bidirectional:
        # _, xs2 = tfp.math.scan_associative(binary_operator,
        #                                   [Lambda_elements, Bu_elements]) # reverse=True
        _, xs2 = tf.scan(tf.vectorized_map(binary_operator), [Lambda_elements, Bu_elements], reverse=True)

        xs = tf.concat((xs, xs2), axis=-1)

    if conj_sym:
        return tf.vectorized_map(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return tf.vectorized_map(lambda x: (C_tilde @ x).real)(xs)


class S5Layer(keras.Model):

    def __init__(self,
                ssm_size,
                blocks,
                H=14,
                discretization='zoh',
                C_init='trunc_standard_normal',
                dt_min=0.001,
                dt_max=0.1,
                conj_sym = True,
                clip_eigs = False,
                bidirectional = False,
                step_rescale = 1.0,
        ):
        super().__init__()
        self.ssm_size = ssm_size
        self.blocks = blocks
        self.H = H
        self.P = ssm_size
        self.C_init = C_init
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale
        self.init_set_up()
        self.setup()

    """ The S5 SSM
        Args:
            
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # TODO initialize parameters here but some are not trainable
        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_im = tf.Variable(self.Lambda_im_init, trainable=True, name="Lambda_im")
        if self.clip_eigs:
            self.Lambda_re = tf.Variable(self.Lambda_re_init, trainable=True,
                                         name="Lambda_re", constraint=lambda x: tf.clip_by_value(x, None, -1e-4))
            self.Lambda = tf.complex(self.Lambda_re, self.Lambda_im)
        else:
            self.Lambda_re = tf.Variable(self.Lambda_re_init, trainable=True, name="Lambda_re")
            self.Lambda = tf.complex(self.Lambda_re, self.Lambda_im)

        # Initialize input to state (B) matrix
        B_shape = (local_P, self.H)
        B = init_VinvB(B_shape, self.Vinv)

        self.B = tf.Variable(B, trainable=True, name='B')
        B_tilde = tf.complex(self.B[..., 0], self.B[..., 1])


        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
        elif self.C_init in ["lecun_normal"]:
            C_init = tf.random.truncated_normal
        elif self.C_init in ["complex_normal"]:
            C_init = tf.random.normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C_shape = [self.H, 2 * self.P, 2]
            else:
                C_shape = [self.H, self.P, 2]
            C = tf.random.normal(shape=C_shape, stddev=0.5 ** 0.5)
            self.C = tf.Variable(C, trainable=True, name='C')
            self.C_tilde = tf.complex(self.C[..., 0], self.C[..., 1])

        else:
            C_shape = (self.H, self.P, 2)
            if self.bidirectional:
                c1 = init_CV(C_init, C_shape, self.V)
                self.C1 = tf.Variable(c1, trainable=True, name='C1')
                c2 = init_CV(C_init, C_shape, self.V)
                self.C2 = tf.Variable(c2, trainable=True, name='C1')

                C1 = tf.complex(self.C1[..., 0], self.C1[..., 1])
                C2 = tf.complex(self.C2[..., 0], self.C2[..., 1])
                self.C_tilde = tf.concat((C1, C2), axis=-1)

            else:
                C = init_CV(C_init, C_shape, self.V)

                self.C = tf.Variable(C, trainable=True, name='C')

                self.C_tilde = tf.complex(self.C[..., 0], self.C[..., 1])

        # Initialize feedthrough (D) matrix
        self.D = tf.Variable(tf.random.normal(shape=[self.H,], stddev=1.0), trainable=True, name='D')

        # TODO Initialize learnable discretization timescale value
        rng_log_steps = tf.random.Generator.get_global_generator().make_seeds(count=1)
        log_step = init_log_steps(key=rng_log_steps, input=(self.P, self.dt_min, self.dt_max))
        self.log_step = tf.Variable(log_step, trainable=True, name='log_step')
        step = self.step_rescale * tf.math.exp(self.log_step[:, 0])


        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def init_set_up(self):
        """
            define:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)

        """
        # determine the size of initial blocks
        block_size = int(self.ssm_size / self.blocks)
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        if self.conj_sym:
            block_size = block_size // 2
            self.P = self.ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        # Vc = V.conj().T
        Vc = tf.transpose(V, conjugate=True)

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * tf.ones((self.blocks, block_size))).ravel()
        self.Lambda_re_init = tf.math.real(Lambda)
        self.Lambda_im_init = tf.math.imag(Lambda)
        self.V = tf.linalg.LinearOperatorBlockDiag(*([V] * self.blocks)) # TODO determine blog_diag
        self.Vinv = tf.linalg.LinearOperatorBlockDiag(*([Vc] * self.blocks))

        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(self.V.shape))
        print("Vinv.shape={}".format(self.Vinv.shape))

    def call(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       input_sequence,
                       self.conj_sym,
                       self.bidirectional)

        # Add feedthrough matrix output Du;
        Du =tf.vectorized_map(lambda u: self.D * u)(input_sequence)
        return ys + Du


import tensorflow as tf
from functools import partial
import tensorflow_probability as tfp
from tensorflow import keras
from einops import rearrange


def my_scan(func):
    def wrapper(L, B):
        return lambda: tf.scan(func, (L, B), parallel_iterations=100)
    return wrapper


def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = tf.math.sqrt(1. + 2. * tf.cast(tf.range(N), tf.float32))
    A = tf.expand_dims(P, -1) * tf.expand_dims(P, 0)
    A = tf.linalg.LinearOperatorLowerTriangular(A).add_to_tensor(-tf.cast(tf.linalg.diag(tf.range(N)), dtype=tf.float32))
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
    P = tf.math.sqrt(tf.cast(tf.range(N), tf.float32) + 0.5)

    # HiPPO also specifies the B matrix
    B = tf.math.sqrt(2.0 * tf.cast(tf.range(N), tf.float32) + 1.0)
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

    S_diag = tf.linalg.diag_part(S)
    Lambda_real = tf.reduce_mean(S_diag) * tf.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = tf.linalg.eigh(tf.complex(0.,-S))
    Lambda_imag = tf.cast(Lambda_imag, Lambda_real.dtype)

    P = tf.transpose(V, conjugate=True) @ tf.cast(P[:, None], tf.complex64)
    B_orig = B
    B = tf.transpose(V, conjugate=True) @ tf.cast(B[:, None], tf.complex64)
    return tf.complex(Lambda_real, Lambda_imag), tf.squeeze(P), tf.squeeze(B), V, B_orig


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(shape):
        """ Init function
             Args:
                 key: random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return tf.random.uniform(shape) * (
            tf.math.log(dt_max) - tf.math.log(dt_min)
        ) + tf.math.log(dt_min)

    return init

# TODO random key in jax
def init_log_steps(input):
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
        # key, skey = tf.random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(shape=(1,))
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
    VinvB = Vinv @ tf.cast(B, Vinv.dtype)
    VinvB_real = tf.math.real(VinvB)
    VinvB_imag = tf.math.imag(VinvB)
    return tf.concat((tf.expand_dims(VinvB_real, -1), tf.expand_dims(VinvB_imag, -1)), axis=-1)


def trunc_standard_normal(shape):
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
        # key, skey = tf.random.split(key) # split seed is not available in stable version
        C = tf.random.truncated_normal(shape=(1, P, 2))
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
    C = tf.complex(C_[..., 0], C_[..., 1])
    # CV = C @ V
    CV = tf.einsum('i j, k j -> i j', C, V)
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
    Identity = tf.ones(Lambda.shape[0], dtype=Lambda.dtype)

    BL = tf.math.divide(1., (Identity - tf.math.divide(Delta, 2.0) * Lambda))
    Lambda_bar = BL * (Identity + tf.math.divide(tf.cast(Delta, Lambda.dtype), 2.0) * Lambda)
    B_bar = tf.expand_dims(BL * tf.cast(Delta, Lambda.dtype), -1) * B_tilde
    Lambda_bar = tf.Variable(Lambda_bar, trainable=True, name='Lambda_bar')
    B_bar = tf.Variable(B_bar, trainable=True, name='B_bar')
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
    Identity = tf.ones(Lambda.shape[0], dtype=Lambda.dtype)
    Lambda_bar = tf.math.exp(Lambda * tf.cast(Delta, Lambda.dtype))
    B_bar = tf.expand_dims(tf.math.divide(1., Lambda) * (Lambda_bar - Identity), -1) * B_tilde
    Lambda_bar = tf.Variable(Lambda_bar, trainable=True, name='Lambda_bar')
    B_bar = tf.Variable(B_bar, trainable=True, name='B_bar')
    return Lambda_bar, B_bar


# Parallel scan operations
# @jax.vmap

# @tf.function
# def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
#     """ Compute the LxH output of discretized SSM given an LxH input.
#         Args:
#             Lambda_bar (complex64): discretized diagonal state matrix    (P,)
#             B_bar      (complex64): discretized input matrix             (P, H)
#             C_tilde    (complex64): output matrix                        (H, P)
#             input_sequence (float32): input sequence of features         (B, L, H)
#             conj_sym (bool):         whether conjugate symmetry is enforced
#             bidirectional (bool):    whether bidirectional setup is used,
#                                   Note for this case C_tilde will have 2P cols
#         Returns:
#             ys (float32): the SSM outputs (S5 layer preactivations)      (B, L, H)
#     """
#     input_sequence = tf.cast(input_sequence, dtype=Lambda_bar.dtype)
#
#     Lambda_elements = Lambda_bar * tf.cast(tf.ones(shape=[tf.shape(input_sequence)[0], tf.shape(input_sequence)[1], Lambda_bar.shape[0]]),
#                                            dtype=Lambda_bar.dtype) # B L P
#     Bu_elements = tf.vectorized_map(lambda u:tf.einsum('p h,l h -> l p', B_bar, u), elems=input_sequence)  # B L P
#     # Bu_elements = tf.vectorized_map(lambda u: B_bar @ u)(input_sequence) #
#
#     # TODO scan associate reverse
#     # _, xs = tfp.math.scan_associative(fn=binary_operator, elems=[Lambda_elements, Bu_elements])
#     _, xs = tf.scan(binary_operator, (Lambda_elements, Bu_elements))
#
#     if bidirectional:
#         # _, xs2 = tfp.math.scan_associative(binary_operator,
#         #                                   [Lambda_elements, Bu_elements]) # reverse=True
#         _, xs2 = tf.scan(binary_operator, (Lambda_elements, Bu_elements), reverse=True)
#
#         xs = tf.concat((xs, xs2), axis=-1)
#     # xs has shape B L P or B L 2P
#     # the return has shape
#     if conj_sym:
#         return tf.vectorized_map(lambda x: 2.0*tf.math.real(tf.einsum('h p, l p -> l h', C_tilde, x)), xs) # h p, b l p -> b l h
#     else:
#         return tf.vectorized_map(lambda x: tf.math.real(tf.einsum('h p, l p -> l h', C_tilde, x)), xs) # h p, b l p -> b l h


class S5Layer(keras.layers.Layer):

    def __init__(self,
                ssm_size=256,
                blocks=8,
                features=64,
                discretization='zoh',
                C_init='lecun_normal',
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
        self.H = features # H
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
            blocks      (int32):     Number of blocks, J, to initialize with 
            channels    (int32):     Number of features of input seq (H in the original codes)
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
            # self.Lambda = tf.Variable(tf.complex(self.Lambda_re, self.Lambda_im),
            #                           trainable=True, name='Lambda',
            #                           constraint=lambda x: tf.clip_by_value(tf.math.real(x), -100000, -1e-4))
        else:
            self.Lambda_re = tf.Variable(self.Lambda_re_init, trainable=True, name="Lambda_re")
            # self.Lambda = tf.Variable(tf.complex(self.Lambda_re, self.Lambda_im),
            #                           dtype=tf.complex64, trainable=True,
            #                           name='Lambda')

        # Initialize input to state (B) matrix
        B_shape = (local_P, self.H)
        B = init_VinvB(B_shape, self.Vinv)

        self.B = tf.Variable(B, trainable=True, name='B')
        # self.B_tilde = tf.Variable(tf.complex(self.B[..., 0], self.B[..., 1]),
        #                            dtype=tf.complex64, trainable=True, name='B_tilde')


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
            self.C = tf.Variable(C, trainable=False, name='C')
            self.C_tilde = tf.Variable(tf.complex(self.C[..., 0], self.C[..., 1]), dtype=tf.complex64, name='C_tilde')

        else:
            C_shape = (self.H, self.P, 2)
            if self.bidirectional:
                c1 = init_CV(C_init, C_shape, self.V)
                self.C1 = tf.Variable(c1, trainable=False, name='C1')
                c2 = init_CV(C_init, C_shape, self.V)
                self.C2 = tf.Variable(c2, trainable=False, name='C1')

                C1 = tf.complex(self.C1[..., 0], self.C1[..., 1])
                C2 = tf.complex(self.C2[..., 0], self.C2[..., 1])
                self.C_tilde = tf.Variable(tf.concat((C1, C2), axis=-1), dtype=tf.complex64, name='C_tilde')

            else:
                # c_rng = tf.random.Generator.make_seeds(count=1)
                C = init_CV(C_init, C_shape, self.V)

                self.C = tf.Variable(C, trainable=True, name='C')

                # self.C_tilde = tf.Variable(tf.complex(self.C[..., 0], self.C[..., 1]), dtype=tf.complex64, name='C_tilde')

        # Initialize feedthrough (D) matrix
        self.D = tf.Variable(tf.random.normal(shape=[self.H,], stddev=1.0), dtype=tf.float32, trainable=True, name='D')

        # TODO Initialize learnable discretization timescale value
        log_step = init_log_steps(input=(self.P, self.dt_min, self.dt_max))
        self.log_step = tf.Variable(log_step, trainable=False, name='log_step')
        self.step = tf.Variable(self.step_rescale * tf.math.exp(self.log_step[:, 0]),
                                dtype=tf.float32, trainable=True, name='Delta')


        # Discretize
        # if self.discretization in ["zoh"]:
        #     # self.discretize = discretize_zoh
        #     self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, self.B_tilde, self.step)
        # elif self.discretization in ["bilinear"]:
        #     # self.discretize = discretize_bilinear
        #     self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, self.B_tilde, self.step)
        # else:
        #     raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

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
        Vc = tf.transpose(V, conjugate=True)
        V = tf.linalg.LinearOperatorFullMatrix(V)
        Vc = tf.linalg.LinearOperatorFullMatrix(Vc)

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = Lambda * tf.cast(tf.ones((self.blocks, block_size)), Lambda.dtype)
        Lambda = tf.reshape(Lambda, [-1])
        self.Lambda_re_init = tf.math.real(Lambda)
        self.Lambda_im_init = tf.math.imag(Lambda)
        self.V = tf.linalg.LinearOperatorBlockDiag([V] * self.blocks).to_dense() # TODO determine blog_diag
        self.Vinv = tf.linalg.LinearOperatorBlockDiag([Vc] * self.blocks).to_dense()

        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(self.V.shape))
        print("Vinv.shape={}".format(self.Vinv.shape))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 100, 64], dtype=tf.float32)])
    def call(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (B, L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (B, L, H)
            Du + ys: output sequence (float32): (B, L, H)
        """

        Identity = tf.ones(tf.shape(self.Lambda_re)[0], dtype=tf.complex64)
        Lambda_bar = tf.complex(tf.exp(self.Lambda_re * self.step) * tf.math.cos(self.Lambda_im * self.step), tf.exp(self.Lambda_re * self.step) * tf.math.sin(self.Lambda_im * self.step))
        B_bar = tf.expand_dims(tf.math.divide(1., tf.complex(self.Lambda_re, self.Lambda_im)) * (Lambda_bar - Identity), -1) * tf.complex(self.B[..., 0], self.B[..., 1])

        ones_elems = tf.cast(tf.ones(shape=[tf.shape(input_sequence)[0], tf.shape(input_sequence)[1], Lambda_bar.shape[0]]),
            dtype=Lambda_bar.dtype)  # B L P
        Lambda_elements = tf.vectorized_map(lambda u: tf.einsum('p, l p -> l p', Lambda_bar, u), ones_elems) # B L P

        Bu_elements = tf.vectorized_map(lambda u: tf.einsum('p h,l h -> l p', B_bar, u),
                                        elems=tf.cast(input_sequence, dtype=Lambda_bar.dtype))  # B L P

        scan_binary_operator = lambda X: tf.scan(self.binary_operator, (X[0], X[1]), parallel_iterations=100)
        _, xs = tf.vectorized_map(scan_binary_operator, (Lambda_elements, Bu_elements))

        if self.bidirectional:
            reversed_scan_binary_operator = lambda X: tf.scan(self.binary_operator, (X[0], X[1]), reverse=True, parallel_iterations=100)
            _, xs2 = tf.vectorized_map(reversed_scan_binary_operator, (Lambda_elements, Bu_elements))

            xs = tf.concat((xs, xs2), axis=-1)
        # xs has shape B L P or B L 2P
        # the return has shape
        C_tilde = tf.complex(self.C[..., 0], self.C[..., 1])
        if self.conj_sym:
            ys = tf.vectorized_map(lambda x: 2.0 * tf.math.real(tf.einsum('h p, l p -> l h', C_tilde, x)),
                                     xs)  # h p, b l p -> b l h
        else:
            ys = tf.vectorized_map(lambda x: tf.math.real(tf.einsum('h p, l p -> l h', C_tilde, x)),
                                     xs)  # h p, b l p -> b l h

        # Add feedthrough matrix output Du;
        Du = tf.vectorized_map(lambda u: self.D * u, input_sequence)
        return ys + Du

    @tf.function
    def binary_operator(self, q_i, q_j):
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

    def built_after_run(self):
        self.built = True

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
# import numpy as np
#
# all_data = np.load('../../datasets/Mujoco/train_mujoco.npy')
# all_data = np.array(all_data)
# training_data = tf.convert_to_tensor(all_data[:64]) # B L K
# # training_data = rearrange(' b l k -> b k l', training_data)
# input_projectstion = diffusion_projection = keras.layers.Dense(64)
# s5_layer = S5Layer()
# # Seq_layer = keras.Sequential()
# # Seq_layer.add(keras.layers.Input(shape=(None, 100, 14), dtype=tf.float32))
# # Seq_layer.add(input_projectstion)
# # Seq_layer.add(s5_layer)
# # Seq_layer(training_data)
# out1 = input_projectstion(training_data)
# out = s5_layer.call(out1)
#

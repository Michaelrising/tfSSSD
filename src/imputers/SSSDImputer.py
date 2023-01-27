import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from utils.util import calc_diffusion_step_embedding
from imputers.S4Model import S4Layer
from imputers.S5Model import S5Layer
from imputers.MegaModel import Mega, MegaLayer

def swish(x):
    return x * keras.activations.sigmoid(x)


class Conv(keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        # TODO only when this is the first layer that needs the input dim
        self.initializer = tf.keras.initializers.HeNormal()
        self.pad = keras.layers.ZeroPadding1D(padding=self.padding)
        self.conv = keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, dilation_rate=dilation,
                                        kernel_initializer=self.initializer, data_format="channels_first")
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = tfa.layers.WeightNormalization(self.conv, data_init=False)
        # nn.init.kaiming_normal_(self.conv.weight)

    @tf.function
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.pad(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        out = self.conv(x)
        return out


class ZeroConv1d(keras.Model):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        # self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.initializer = tf.keras.initializers.Zeros()
        self.conv = keras.layers.Conv1D(filters=out_channel, kernel_size=1,
                                        kernel_initializer=self.initializer,
                                        bias_initializer=self.initializer, data_format="channels_first")

    @tf.function
    def call(self, x):
        out = self.conv(x)
        return out


class Residual_block(keras.Model):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 alg,
                 ):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        # self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        self.fc_t = keras.layers.Dense(self.res_channels)
        if alg == 'S4':
            self.SSM1 = S4Layer(features=2 * self.res_channels,
                               lmax=s4_lmax,
                               N=s4_d_state,
                               dropout=s4_dropout,
                               bidirectional=s4_bidirectional,
                               layer_norm=s4_layernorm,
                               )
            self.SSM2 = S4Layer(features=2 * self.res_channels,
                               lmax=s4_lmax,
                               N=s4_d_state,
                               dropout=s4_dropout,
                               bidirectional=s4_bidirectional,
                               layer_norm=s4_layernorm,
                               )
        elif alg == 'S5':
            self.SSM1 = S5Layer(ssm_size=s4_d_state, features=2 * self.res_channels) # ssm_size has Order(H)
            self.SSM2 = S5Layer(ssm_size=s4_d_state, features=2 * self.res_channels)
        elif alg == 'Mega':
            self.SSM1 = MegaLayer(features=2 * self.res_channels, chunk_size=-1)
            self.SSM2 = MegaLayer(features=2 * self.res_channels, chunk_size=-1)

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.cond_conv = Conv(2 * in_channels, 2 * self.res_channels, kernel_size=1)

        self.res_conv = Conv(in_channels, res_channels, 1)

        self.skip_conv = Conv(in_channels, skip_channels, 1)

    @tf.function
    def call(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.expand_dims(part_t, -1)
        h = h + part_t

        h = self.conv_layer(h)
        h = self.SSM1(tf.transpose(h, perm=[0, 2, 1])) # SSM input shape is B L C output has shape B L C
        h = tf.transpose(h, perm=[0, 2, 1]) # B C L

        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond

        h = self.SSM2(tf.transpose(h, perm=[0, 2, 1])) # SSM input shape is B L C
        h = tf.transpose(h, perm=[0, 2, 1]) # B C L

        out = tf.math.tanh(h[:, :self.res_channels, :]) * tf.math.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(keras.Model):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 alg,
                 ):
        super(Residual_group, self).__init__()

        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_model = keras.Sequential()
        self.fc_model.add(keras.layers.Input((diffusion_step_embed_dim_in,)))
        self.fc_model.add(keras.layers.Dense(diffusion_step_embed_dim_mid, activation=swish))
        self.fc_model.add(keras.layers.Dense(diffusion_step_embed_dim_out, activation=swish))

        # self.residual_blocks = nn.ModuleList()
        self.residual_blocks = []
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm,
                                                       alg=alg,
                                                       ))

    @tf.function
    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        # diffusion_step_embed = tf.transpose(diffusion_step_embed, perm= [1, 0]) # change dimension seq and feature, feature to last
        diffusion_step_embed = self.fc_model(diffusion_step_embed)

        # diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSDImputer(keras.Model):
    def __init__(self,
                 in_channels,
                 res_channels,
                 skip_channels,
                 out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 alg):
        super(SSSDImputer, self).__init__()

        self.init_conv = keras.Sequential()  # initial process for input
        self.init_conv.add(keras.layers.Input(shape=(in_channels, None,)))
        self.init_conv.add(Conv(in_channels, res_channels, kernel_size=1))
        self.init_conv.add(keras.layers.Activation(keras.activations.relu))

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm,
                                             alg=alg,
                                             )

        self.final_conv = keras.Sequential()
        self.final_conv.add(keras.layers.Input(shape=(skip_channels, None,)))
        self.final_conv.add(Conv(in_channels=skip_channels, out_channels=skip_channels, kernel_size=1))
        self.final_conv.add(keras.layers.Activation(keras.activations.relu))
        self.final_conv.add(ZeroConv1d(skip_channels, out_channels))

    @tf.function
    def call(self, input_data, training=True):
        noise, conditional, mask, diffusion_steps = input_data

        conditional = conditional * mask
        conditional = tf.concat([conditional, mask], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y

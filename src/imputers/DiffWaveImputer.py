import math
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from utils.util import calc_diffusion_step_embedding


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

    def call(self, x):
        out = self.conv(x)
        return out


class Residual_block(keras.Model):
    def __init__(self, res_channels, skip_channels, dilation, 
                 diffusion_step_embed_dim_out, in_channels):
        super(Residual_block, self).__init__()
        
        self.res_channels = res_channels
        # the layer-specific fc for diffusion step embedding
        self.fc_t = keras.layers.Dense(self.res_channels)
        
        # dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)
        
        # add mel spectrogram upsampler and conditioner conv1x1 layer  (In adapted to S4 output)
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)  # 80 is mel bands

        self.res_conv = Conv(in_channels, res_channels, 1)

        self.skip_conv = Conv(in_channels, skip_channels, 1)

    def call(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels                      
                                                           
        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t, [B, self.res_channels, 1])
        h = h + part_t
        
        h = self.dilated_conv_layer.call(h)
        # add (local) conditioner
        assert cond is not None

        cond = self.cond_conv.call(cond)
        h += cond
        
        out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv.call(out)
        assert x.shape == res.shape
        skip = self.skip_conv.call(out)

        return (x + res) * math.sqrt(0.5), skip  

    
    

class Residual_group(keras.Model):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 device):
        super(Residual_group, self).__init__()
        self.device = device
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = keras.layers.Dense(diffusion_step_embed_dim_out)

        self.residual_blocks = []
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       dilation=2 ** (n % dilation_cycle),
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels))

    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in, self.device)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for block in self.residual_blocks:
            h, skip_n = block.call((noise, conditional, diffusion_step_embed))
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  # normalize for training stability


class DiffWaveImputer(keras.Model):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers, dilation_cycle, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 device):
        super(DiffWaveImputer, self).__init__()
        self.device = device

        self.init_conv = keras.layers.Conv1D(res_channels, kernel_size=1, activation='relu', data_format="channels_first")

        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             dilation_cycle=dilation_cycle,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             device=device)

        self.final_conv = keras.Sequential()
        self.final_conv.add(keras.layers.Input(shape=(skip_channels, None,)))
        self.final_conv.add(Conv(in_channels=skip_channels, out_channels=skip_channels, kernel_size=1))
        self.final_conv.add(keras.layers.Activation(keras.activations.relu))
        self.final_conv.add(ZeroConv1d(skip_channels, out_channels))

    def call(self, input_data):

        noise, conditional, mask, diffusion_steps = input_data
       
        conditional = conditional * mask
        conditional = tf.concat([conditional, mask], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer.call((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y


import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from utils.util import calc_diffusion_step_embedding
from imputers.S4Model import S4Layer


def swish(x):
    return x * keras.activations.sigmoid(x)


class Conv(keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        # TODO only when this is the first layer that needs the input dim
        self.initializer = tf.keras.initializers.HeNormal()
        self.pad = keras.layers.ZeroPadding2D(paddings=((0,0), (self.padding, self.padding)))
        self.conv = keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, dilation_rate=dilation, input_shape=in_channels, kernel_initializer=self.initializer)
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = tfa.layers.WeightNormalization(self.conv)
        # nn.init.kaiming_normal_(self.conv.weight)

    def call(self, x):
        x = self.pad(x)
        out = self.conv(x)
        return out
    
    
class ZeroConv1d(keras.Model):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        # self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.initializer = tf.keras.initializers.Zeros()
        self.conv = keras.layers.Conv1D(filters=out_channel, kernel_size=1,
                                        input_shape=in_channel,  kernel_initializer=self.initializer, bias_initializer=self.initializer)

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
                 device):
        super(Residual_block, self).__init__()
        self.device = device
        self.res_channels = res_channels


        # self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        self.fc_t = keras.layers.Dense( self.res_channels)
        
        self.S41 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm,
                          device = device)
 
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm,
                          device = device)
        
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)

        initializer = tf.keras.initializers.HeNormal()
        self.res_conv = keras.layers.Conv1D(filters=res_channels, kernel_size=1, kenerl_initializer=initializer)
        self.res_conv = tfa.layers.WeightNormalization(self.res_conv)
        # nn.init.kaiming_normal_(self.res_conv.weight)

        
        self.skip_conv = keras.layers.Conv1D(filers=skip_channels, kernel_size=1, kenerl_initializer=initializer)
        self.skip_conv =  tfa.layers.WeightNormalization(self.skip_conv)
        self.permute0 = tf.keras.layers.Permute((2, 0, 1))
        self.permute1 = tf.keras.layers.Permute((1, 2, 0))
        # nn.init.kaiming_normal_(self.skip_conv.weight)

    def call(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t, [B, self.res_channels, 1])
        h = h + part_t
        
        h = self.conv_layer.call(h)
        h = self.S41(self.permute0(h))
        h = self.permute1(h)
        
        assert cond is not None
        cond = self.cond_conv.call(cond)
        h += cond
        
        h = self.S42(self.permute0(h))
        h = self.permute1(h)
        
        out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])

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
                 device):
        super(Residual_group, self).__init__()
        self.device = device
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.input_layer = keras.layers.Input(shape = (diffusion_step_embed_dim_in,))
        self.fc_t1 = keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = keras.layers.Dense(diffusion_step_embed_dim_out)
        
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
                                                       device=device))

            
    def forward(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n].call((h, conditional, diffusion_step_embed))
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  


class SSSDS4Imputer(keras.Model):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 device):
        super(SSSDS4Imputer, self).__init__()
        self.device = device
        self.init_conv = keras.layers.Sequential(Conv(in_channels, res_channels, kernel_size=1), keras.layers.Activation(keras.activations.relu)) # initial process for input
        
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
                                             device = device)
        
        self.final_conv = keras.layers.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        keras.layers.Activation(keras.activations.relu),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):
        
        noise, conditional, mask, diffusion_steps = input_data 

        conditional = conditional * mask
        conditional = tf.concat([conditional, tf.cast(mask,dtype = tf.float32)], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer.call((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from einops import rearrange
from imputers.S4Model import S4, LinearActivation
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


class DownPool(keras.Model):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
            weight_norm=True,
        )

    def call(self, x):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(tf.stack(state, axis=-1), '... h s -> ... (h s)')
            x = tf.expand_dims(x, -1)
            x = self.linear(x)
            x = tf.squeeze(x, -1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


class UpPool(keras.Model):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
            weight_norm=True,
        )

    def call(self, x):
        x = self.linear(x)
        
        if(self.causal):
            padding = [[0, 0] * len(x[..., :-1].shape)]
            padding[-1] = [1,0]
            x = tf.pad(x[..., :-1], padding) # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            x = tf.expand_dims(x, -1)
            x = self.linear(x)
            x = tf.squeeze(x, -1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(tf.unstack(x, axis=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = tf.zeros(batch_shape + (self.d_output, self.pool)) # (batch, h, s)
        state = list(tf.unstack(state, axis=-1)) # List of (..., H)
        return state


class FFBlock(keras.Model):

    def __init__(self, d_model, expand=2, dropout=0.0):
        """
        Feed-forward block.

        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()

        input_linear = LinearActivation(
            d_model, 
            d_model * expand,
            transposed=True,
            activation='gelu',
            activate=True,
        )
        dropout = keras.layers.SpatialDropout2D(dropout) if dropout > 0.0 else tf.identity
        output_linear = LinearActivation(
            d_model * expand,
            d_model, 
            transposed=True,
            activation=None,
            activate=False,
        )

        self.ff = keras.Sequential(
            input_linear,
            dropout,
            output_linear,
        )

    def call(self, x):
        return self.ff(x), None

    def default_state(self, *args, **kwargs):
        return None

    def step(self, x, state, **kwargs):
        # expects: (B, D, L)
        return tf.squeeze(self.ff(tf.expand_dims(x, -1)),-1), state


class ResidualBlock(keras.Model):

    def __init__(
        self, 
        d_model, 
        layer,
        dropout,
        diffusion_step_embed_dim_out,
        in_channels,
        label_embed_dim,
        stride
    ):
        
        """
        Residual S4 block.

        Args:
            d_model: dimension of the model
            bidirectional: use bidirectional S4 layer
            glu: use gated linear unit in the S4 layer
            dropout: dropout rate
        """
        super().__init__()

        self.layer = layer
        self.norm = keras.layers.LayerNormalization(axis=-1) #nn.LayerNorm(d_model)
        self.dropout = keras.layers.SpatialDropout2D(dropout) if dropout > 0.0 else tf.identity

        self.fc_t = keras.layers.Dense(d_model) # diffusion_step_embed_dim_out,
        self.cond_conv = Conv(2*in_channels, d_model, kernel_size=stride, stride=stride)
        self.fc_label = keras.layers.Dense(d_model)  if label_embed_dim is not None else None
        
        
    def call(self, input_data):
        """
        Input x is shape (B, d_input, L)
        """
        x, cond, diffusion_step_embed = input_data
        
        # add in diffusion step embedding
        part_t = tf.expand_dims(self.fc_t(diffusion_step_embed), 2)
        z = x + part_t
        
        # Prenorm
        z = self.norm(tf.transpose(z, [-1, -2])) #.transpose(-1, -2)
        z = tf.transpose(z, [-1,-2])
        
        z, _ = self.layer(z)
        
        cond = self.cond_conv.call(cond)
        #cond = self.fc_label(cond)
      
    
        z = z + cond
            
        # Dropout on the output of the layer
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x

    
    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x

        # Prenorm
        z = self.norm(z)

        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)

        # Residual connection
        x = z + x

        return x, state


class SSSDSAImputer(keras.Model):
    def __init__(
        self,
        d_model=128, 
        n_layers=6, 
        pool=[2, 2], 
        expand=2, 
        ff=2, 
        glu=True,
        unet=True,
        dropout=0.0,
        in_channels=1,
        out_channels=1,
        diffusion_step_embed_dim_in=128, 
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
        label_embed_dim=128,
        label_embed_classes=71,
        bidirectional=True,
        s4_lmax=1,
        s4_d_state=64,
        s4_dropout=0.0,
        s4_bidirectional=True,
    ):
        
        """
        SaShiMi model backbone. 

        Args:
            d_model: dimension of the model. We generally use 64 for all our experiments.
            n_layers: number of (Residual (S4) --> Residual (FF)) blocks at each pooling level. 
                We use 8 layers for our experiments, although we found that increasing layers even further generally 
                improves performance at the expense of training / inference speed.
            pool: pooling factor at each level. Pooling shrinks the sequence length at lower levels. 
                We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
                It's possible that a different combination of pooling factors and number of tiers may perform better.
            expand: expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
                We generally found 2 to perform best (among 2, 4).
            ff: expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4).
            bidirectional: use bidirectional S4 layers. Bidirectional layers are suitable for use with non-causal models 
                such as diffusion models like DiffWave.
            glu: use gated linear unit in the S4 layers. Adds parameters and generally improves performance.
            unet: use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling. 
                All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
                We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
                for non-autoregressive models.
            dropout: dropout rate. Default to 0.0, since we haven't found settings where SaShiMi overfits.
        """
        super().__init__()
        self.d_model = H = d_model
        self.unet = unet

        def s4_block(dim, stride):
          
            layer = S4(
                d_model=dim, 
                l_max=s4_lmax,
                d_state=s4_d_state,
                bidirectional=s4_bidirectional,
                postact='glu' if glu else None,
                dropout=dropout,
                transposed=True,
                #hurwitz=True, # use the Hurwitz parameterization for stability
                #tie_state=True, # tie SSM parameters across d_state in the S4 layer
                trainable={
                    'dt': True,
                    'A': True,
                    'P': True,
                    'B': True,
                }, # train all internal S4 parameters
                    
            )
            
                
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels = in_channels,
                label_embed_dim = label_embed_dim,
                stride=stride     
            )

        def ff_block(dim, stride):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels = in_channels,
                label_embed_dim = label_embed_dim,
                stride=stride
            )

        # Down blocks
        d_layers = []
        for i, p in enumerate(pool):
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    if i == 0:
                        d_layers.append(s4_block(H, 1))
                        if ff > 0: d_layers.append(ff_block(H, 1))
                    elif i == 1:
                        d_layers.append(s4_block(H, p))
                        if ff > 0: d_layers.append(ff_block(H, p))
            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand
        
        # Center block
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(s4_block(H, pool[1]*2))
            if ff > 0: c_layers.append(ff_block(H, pool[1]*2))
        
        # Up blocks
        u_layers = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p, causal= not bidirectional))

            for _ in range(n_layers):
                if i == 0:
                    block.append(s4_block(H, pool[0]))
                    if ff > 0: block.append(ff_block(H, pool[0]))
                        
                elif i == 1:
                    block.append(s4_block(H, 1))
                    if ff > 0: block.append(ff_block(H, 1))

            u_layers.append(block)
        
        self.d_layers = d_layers # nn.ModuleList(d_layers)
        self.c_layers = c_layers # nn.ModuleList(c_layers)
        self.u_layers = u_layers # nn.ModuleList(u_layers)
        self.norm = keras.layers.LayerNormalization(axis=-1) #nn.LayerNorm(H)

        self.init_conv = keras.layers.Conv1D(d_model, kernel_size=1, activation='relu', data_format="channels_first") #nn.Sequential(nn.Conv1d(in_channels,d_model,kernel_size=1),nn.ReLU())
        self.final_conv = keras.Sequential() #nn.Sequential(nn.Conv1d(d_model,d_model,kernel_size=1),nn.ReLU(),nn.Conv1d(d_model,out_channels,kernel_size=1))
        self.final_conv.add(keras.layers.Conv1D(d_model, kernel_size=1, activation='relu', data_format="channels_first"))
        self.final_conv.add(keras.layers.Conv1D(out_channels, kernel_size=1, data_format="channels_first"))
        self.fc_t1 = keras.layers.Dense(diffusion_step_embed_dim_mid) #nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = keras.layers.Dense(diffusion_step_embed_dim_out) #nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        self.cond_embedding = keras.layers.Embedding(label_embed_classes, label_embed_dim) if label_embed_classes>0 != None else None
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        assert H == d_model

    def call(self, input_data):
        
        # (transformed_X, cond, mask, diffusion_steps.view(B,1),))
        #audio_cond: same shape as audio, audio_mask: same shape as audio but binary to be imputed where zero
        noise, conditional, mask, diffusion_steps = input_data 

        conditional = conditional * mask       
        conditional = tf.concat([conditional, mask.float()],axis=1)

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        x = noise        
        x = self.init_conv(x)   

        # Down blocks
        outputs = []
        outputs.append(x)
        for layer in self.d_layers:
            if isinstance(layer, ResidualBlock):
                x = layer.call((x,conditional,diffusion_step_embed))
            else:
                x = layer.call(x)
            outputs.append(x)
            
        # Center block
        for layer in self.c_layers:
            if isinstance(layer, ResidualBlock):
                x = layer.call((x,conditional,diffusion_step_embed))
            else:
                x = layer.call(x)
        x = x + outputs.pop() # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer.call((x,conditional,diffusion_step_embed))
                    else:
                        x = layer.call(x)
                    x = x + outputs.pop() # skip connection
            else:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer.call((x,conditional,diffusion_step_embed))
                    else:
                        x = layer.call(x)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block

        # feature projection
        x = tf.transpose(x, [0, 1, 2]) # (batch, length, expand)
        x = self.norm(x)
        x = tf.transpose(x, [0, 1, 2]) # (batch, expand, length)
        x = self.final_conv(x) # 128 to 12
        return x 

    def default_state(self, *args, **kwargs):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SaShiMi
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None: break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                for i in range(skipped):
                    next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped//3:]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            if self.unet:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    x = x + outputs.pop()
            else:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()

        # feature projection
        x = self.norm(x)
        return x, next_state

    def setup_rnn(self, mode='dense'):
        """
        Convert the SaShiMi model to a RNN for autoregressive generation.

        Args:
            mode: S4 recurrence mode. Using `diagonal` can speed up generation by 10-20%. 
                `linear` should be faster theoretically but is slow in practice since it 
                dispatches more operations (could benefit from fused operations).
                Note that `diagonal` could potentially be unstable if the diagonalization is numerically unstable
                (although we haven't encountered this case in practice), while `dense` should always be stable.
        """
        assert mode in ['dense', 'diagonal', 'linear']
        for module in self.modules():
            if hasattr(module, 'setup_step'): module.setup_step(mode)


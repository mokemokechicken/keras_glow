from logging import getLogger

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K, initializers, Input
from tensorflow.python.keras.engine import Layer, Network, base_layer
from tensorflow.python.keras.layers import Conv2D, Activation


logger = getLogger(__name__)


class PreProcess(Layer):
    """for easy to serialization"""

    def __init__(self, n_bins=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return K.cast(inputs, 'float32') / self.n_bins - 0.5

    def get_config(self):
        base_config = super().get_config()
        config = {
            'n_bins': self.n_bins,
        }
        config.update(base_config)
        return config


class PostProcess(Layer):
    """for easy to serialization"""

    def __init__(self, n_bins=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return K.cast(K.clip(tf.floor((inputs + 0.5) * self.n_bins), 0, 255), 'uint8')

    def get_config(self):
        base_config = super().get_config()
        config = {
            'n_bins': self.n_bins,
        }
        config.update(base_config)
        return config


class AddRandomUniform(Layer):
    def __init__(self, n_bins=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return inputs+tf.random_uniform(tf.shape(inputs), 0, 1. / self.n_bins)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'n_bins': self.n_bins,
        }
        config.update(base_config)
        return config


class ActNorm(Layer):
    log_scale = None
    bias = None

    def __init__(self, bit_per_sub_pixel_factor=None, use_loss=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_loss = use_loss
        self.bit_per_sub_pixel_factor = bit_per_sub_pixel_factor

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        n_ch = input_shape[-1]
        assert len(input_shape) in (2, 4), f'invalid input_shape={input_shape}'

        if len(input_shape) == 2:
            var_shape = (1, n_ch)
            log_det_factor = 1
        else:
            var_shape = (1, 1, 1, n_ch)
            log_det_factor = int(input_shape[1]*input_shape[2])

        # DDI(Data-Dependent-Init?) is not implemented
        self.log_scale = self.add_weight('log_scale', shape=var_shape, initializer='zeros')
        self.bias = self.add_weight('bias', shape=var_shape, initializer='zeros')

        if self.use_loss:
            # Log-Determinant
            # it seems that this is required only for encoding.
            self.add_loss(-1 * log_det_factor * K.sum(self.log_scale) * self.bit_per_sub_pixel_factor)
            # K.sum(self.log_scale) or K.sum(K.abs(self.log_scale)) ???

        # final
        super().build(input_shape)

    def call(self, inputs, reverse=False, **kwargs):
        x = inputs
        if not reverse:
            return (x + self.bias) * K.exp(self.log_scale)
        else:
            return x / K.exp(self.log_scale) - self.bias

    def get_config(self):
        base_config = super().get_config()
        config = {
            'bit_per_sub_pixel_factor': self.bit_per_sub_pixel_factor,
        }
        config.update(base_config)
        return config

    def get_log_scale(self):
        return K.get_value(self.log_scale)


class Invertible1x1Conv(Layer):
    rotate_matrix = None  # type: tf.Variable
    determinant = None  # type: tf.Variable

    def __init__(self, bit_per_sub_pixel_factor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bit_per_sub_pixel_factor = bit_per_sub_pixel_factor

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        w_shape = [input_shape[3], input_shape[3]]  # [n_channel, n_channel]

        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        self.rotate_matrix = self.add_weight("rotate_matrix", w_shape, initializer=initializers.constant(w_init),
                                             trainable=True)
        # debug
        self.determinant = K.variable(value=np.linalg.det(w_init), name='determinant')

        # add log-det as loss
        log_det_factor = int(input_shape[1] * input_shape[2])
        # log_det = tf.log(tf.abs(tf.matrix_determinant(self.rotate_matrix)))
        # log_det = tf.log(tf.abs(tf.matrix_determinant(self.rotate_matrix)) + K.epsilon())
        log_det = tf.log(tf.clip_by_value(tf.abs(tf.matrix_determinant(self.rotate_matrix)), 0.001, 1000))
        self.add_loss(-1 * log_det_factor * log_det * self.bit_per_sub_pixel_factor)

        self.add_update([K.update(self.determinant, tf.matrix_determinant(self.rotate_matrix))])
        # final
        super().build(input_shape)

    def call(self, inputs, reverse=False, **kwargs):
        w = self.rotate_matrix
        if reverse:
            w = tf.matrix_inverse(w)
        w = tf.expand_dims(tf.expand_dims(w, axis=0), axis=0)
        z = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
        return z

    def get_config(self):
        base_config = super().get_config()
        config = {
            'bit_per_sub_pixel_factor': self.bit_per_sub_pixel_factor,
        }
        config.update(base_config)
        return config

    def get_determinant(self):
        return K.get_value(self.determinant)


class AffineCoupling(Network):  # FlowCoupling
    def __init__(self, in_shape=None, hidden_channel_size=None, bit_per_sub_pixel_factor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_shape = in_shape
        self.hidden_channel_size = hidden_channel_size
        self.bit_per_sub_pixel_factor = bit_per_sub_pixel_factor
        self.conv1 = Conv2D(filters=self.hidden_channel_size, name=f'{self.name}/conv1',
                            kernel_size=3, strides=1, padding="same", use_bias=False)
        self.actnorm1 = ActNorm(use_loss=False, name=f'{self.name}/actnorm1')

        self.conv2 = Conv2D(filters=self.hidden_channel_size, name=f'{self.name}/conv2',
                            kernel_size=1, strides=1, padding="same", use_bias=False)
        self.actnorm2 = ActNorm(use_loss=False, name=f'{self.name}/actnorm2')

        self.last_conv = Conv2D(filters=self.in_shape[-1], kernel_size=3, padding="same", name=f'{self.name}/last_conv',
                                kernel_initializer='zero', bias_initializer='zero')

        # ------------ monkey-patch -----------------
        # (1) Avoid error in __call__()
        self.outputs = []
        # (2) Avoid Model.get_config() -> from_config() infinite loop (by pushing dummy node)
        # Create the node linking internal inputs to internal outputs.
        self.in_x = Input(shape=self.in_shape)
        base_layer.Node(
            outbound_layer=self,
            inbound_layers=[],
            node_indices=[],
            tensor_indices=[],
            input_tensors=[self.in_x],
            output_tensors=self.outputs)

    def call(self, inputs, reverse=False, **kwargs):
        z = inputs
        z1, z2 = split_channels(z)

        scale, shift = split_channels(self.nn(z1))
        # scale = K.exp(scale)  # seems not stable to train
        scale = 1 + K.tanh(scale) * 0.2  # how about this?
        # scale = K.sigmoid(scale + 2)  # ?? from reference implementation
        if not reverse:
            z2 = (z2 + shift) * scale
            self.add_loss(-K.sum(K.log(scale), axis=[1, 2, 3]) * self.bit_per_sub_pixel_factor)
        else:
            z2 = z2 / scale - shift
        out = K.concatenate([z1, z2], axis=3)
        return out

    def nn(self, out):
        """n_ch of output is same as n_ch of input_shape"""
        out = self.conv1(out)
        out = self.actnorm1(out)
        out = Activation('relu')(out)

        out = self.conv2(out)
        out = self.actnorm2(out)
        out = Activation('relu')(out)

        out = self.last_conv(out)
        return out

    def get_config(self):
        base_config = super(Network, self).get_config()  # Network.get_config() is not implemented
        config = {
            'name': base_config.get('name'),
            'in_shape': self.in_shape,
            'hidden_channel_size': self.hidden_channel_size,
            'bit_per_sub_pixel_factor': self.bit_per_sub_pixel_factor,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # logger.debug(f'called={config}, custom={custom_objects}')
        return cls(**config)


class Squeeze2d(Layer):
    factor = 2

    def compute_output_shape(self, input_shape):
        assert input_shape[1] % self.factor == 0 and input_shape[2] % self.factor == 0, f'{input_shape}, {self.factor}'
        return [input_shape[0],
                input_shape[1]//self.factor, input_shape[2]//self.factor,
                input_shape[3]*self.factor*self.factor]

    def call(self, inputs, **kwargs):
        x = inputs
        _, height, width, n_ch = K.int_shape(x)
        factor = self.factor
        x = K.reshape(x, [-1, height // factor, factor, width // factor, factor, n_ch])
        x = K.permute_dimensions(x, [0, 1, 3, 5, 2, 4])
        x = K.reshape(x, [-1, height // factor, width // factor, n_ch * factor * factor])
        return x


class Unsqueeze2d(Layer):
    factor = 2

    def compute_output_shape(self, input_shape):
        bs, height, width, n_ch = input_shape
        factor = self.factor
        assert n_ch >= factor**2 and n_ch % factor**2 == 0, f'n_ch={n_ch}, input_shape={input_shape}'
        return [bs, height*factor, width*factor, n_ch//(factor**2)]

    def call(self, inputs, **kwargs):
        x = inputs
        factor = self.factor
        bs, height, width, n_ch = K.int_shape(inputs)
        assert n_ch >= factor**2 and n_ch % factor**2 == 0, f'n_ch={n_ch}, input_shape={K.int_shape(inputs)}'
        x = K.reshape(x, [-1, height, width, n_ch//(factor**2), factor, factor])
        x = K.permute_dimensions(x, [0, 1, 4, 2, 5, 3])
        x = K.reshape(x, [-1, height*factor, width*factor, n_ch//(factor**2)])
        return x


class Split2d(Network):
    def __init__(self, in_shape=None, bit_per_sub_pixel_factor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_shape = in_shape
        self.bit_per_sub_pixel_factor = bit_per_sub_pixel_factor
        self.conv = Conv2D(filters=self.in_shape[-1], kernel_size=3, padding="same",
                           kernel_initializer='zero', bias_initializer='zero')

        # ------------ monkey-patch -----------------
        # (1) Avoid error in __call__()
        self.outputs = []
        # (2) Avoid Model.get_config() -> from_config() infinite loop (by pushing dummy node)
        # Create the node linking internal inputs to internal outputs.
        self.in_x = Input(shape=self.in_shape)
        base_layer.Node(
            outbound_layer=self,
            inbound_layers=[],
            node_indices=[],
            tensor_indices=[],
            input_tensors=[self.in_x],
            output_tensors=self.outputs)

    def call(self, inputs, reverse=False, **kwargs):
        if not reverse:
            z1, z2 = split_channels(inputs)  # (w, h, n_ch//2)

            # split2d_prior(z)
            h = self.conv(z1)  # (w, h, n_ch)
            pz = GaussianDiag(h)  # (w, h, n_ch//2)
            # out = Squeeze2d()(z1)  # (w//2, h//2, n_ch*2)  move to encoder_loop() for corresponding to the paper
            self.add_loss(-1 * pz.logp(z2) * self.bit_per_sub_pixel_factor)
            out = z1
        else:
            # z1 = Unsqueeze2d()(inputs)  # (w, h, n_ch//2)  # move to decoder_loop() for corresponding to the paper
            z1, temperature = inputs
            h = self.conv(z1)  # (w, h, n_ch)
            pz = GaussianDiag(h)  # (w, h, n_ch//2)
            z2 = pz.sample2(pz.eps * K.reshape(temperature, [-1, 1, 1, 1]))
            out = K.concatenate([z1, z2], axis=3)
        return out

    def get_config(self):
        base_config = super(Network, self).get_config()  # Network.get_config() is not implemented
        config = {
            'name': base_config.get('name'),
            'in_shape': self.in_shape,
            'bit_per_sub_pixel_factor': self.bit_per_sub_pixel_factor,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # logger.debug(f'called={config}, custom={custom_objects}')
        return cls(**config)


class JustForAddLoss(Layer):
    def __init__(self, constant_loss=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant_loss = constant_loss

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.add_loss(self.constant_loss)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'constant_loss': self.constant_loss,
        }
        config.update(base_config)
        return config


class GaussianDiag:
    def __init__(self, tensor):
        self.mean, self.logsd = split_channels(tensor)
        self.eps = K.random_normal(K.shape(self.mean))  # eps means like temperature
        self.sample = self.mean + K.exp(self.logsd) * self.eps

    @classmethod
    def prior(cls, shape):
        tensor = K.concatenate([K.zeros(shape), K.zeros(shape)], axis=-1)
        return cls(tensor)

    def sample2(self, eps):
        return self.mean + K.exp(self.logsd) * eps

    def logps(self, x):
        return -0.5 * (np.log(2 * np.pi) + 2. * self.logsd + (x - self.mean) ** 2 / K.exp(2. * self.logsd))

    def logp(self, x):
        return K.sum(self.logps(x), axis=list(range(K.ndim(x)))[1:])


def split_channels(tensor):
    n_ch = K.int_shape(tensor)[-1]
    assert K.ndim(tensor) in [2, 4], f'tensor shape={K.int_shape(tensor)}'

    if K.ndim(tensor) == 2:
        return tensor[:, :n_ch // 2], tensor[:, n_ch // 2:]
    elif K.ndim(tensor) == 4:
        return tensor[:, :, :, :n_ch // 2], tensor[:, :, :, n_ch // 2:]
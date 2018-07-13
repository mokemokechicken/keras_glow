from logging import getLogger

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras import constraints
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import Layer, Network
from tensorflow.python.keras.layers import Lambda, Conv2D, Activation
from tensorflow.python.keras.models import load_model

from keras_glow.config import Config

logger = getLogger(__name__)


class GlowModel:
    def __init__(self, config: Config):
        self.config = config
        self._layers = {}
        self.encoder = None  # type: Model
        self.decoder = None  # type: Model
        self.bit_per_sub_pixel_factor = None  # type: float

    def build(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def save_all(self):
        rc = self.config.resource
        logger.info(f"saving encoder to {rc.encoder_path}")
        self.encoder.save(rc.encoder_path)
        logger.info(f"saving decoder to {rc.decoder_path}")
        self.decoder.save(rc.decoder_path)

    def load_all(self):
        rc = self.config.resource
        logger.info(f"loading encoder from {rc.encoder_path}")
        self.encoder = load_model(rc.encoder_path)
        logger.info(f"loading decoder from {rc.decoder_path}")
        self.decoder = load_model(rc.decoder_path)

    def build_encoder(self):
        mc = self.config.model
        dc = self.config.data
        logger.info(mc)
        in_shape = (dc.image_height, dc.image_width, 3)
        in_x = Input(shape=in_shape, name='image', dtype='uint8')

        # for loss to bits per sub pixel
        self.bit_per_sub_pixel_factor = 1. / (np.log(2.) * np.prod(in_shape))
        logger.debug(f'bit_per_sub_pixel_factor={self.bit_per_sub_pixel_factor}')

        # pre-process
        out = Lambda(lambda x: K.cast(x, 'float32') / mc.n_bins - 0.5, name="pre-process")(in_x)
        # add noise
        out = Lambda(lambda x: x+tf.random_uniform(tf.shape(x), 0, 1. / mc.n_bins), name="add_random_uniform")(out)

        # encoder_loop
        encoder_loop_out = self.build_encoder_loop(out)
        encoder = Model(inputs=in_x, outputs=encoder_loop_out)

        # add prior loss
        prior = GaussianDiag.prior(K.shape(encoder_loop_out))
        encoder.add_loss(-prior.logp(encoder_loop_out) * self.bit_per_sub_pixel_factor)

        # `objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z)[1:])`
        encoder.add_loss(np.log(mc.n_bins) * np.prod(in_shape) * self.bit_per_sub_pixel_factor)
        return encoder

    def build_encoder_loop(self, out):
        mc = self.config.model
        for level_idx in range(mc.n_levels):
            out = Squeeze2d()(out)
            out = self.revnet2d(out, level_idx)  # 'step of flow' in the paper
            if level_idx < mc.n_levels - 1:
                out = self.split2d(out, level_idx)
        return out

    def build_decoder(self, z_shape=None):
        mc = self.config.model
        z_shape = z_shape or K.int_shape(self.encoder.output)[1:]

        # Placeholder
        z_in = Input(shape=z_shape, name="z_in")
        temperature = Input(shape=(1, ), name="temperature")

        # build_decoder_loop
        out = z_in
        for level_idx in reversed(range(mc.n_levels)):
            if level_idx < mc.n_levels - 1:
                out = self.split2d_reverse(out, temperature, level_idx)
            out = self.revnet2d(out, level_idx, reverse=True)
            out = Unsqueeze2d()(out)
        # post-process
        out = Lambda(lambda x: K.cast(K.clip(tf.floor((x + 0.5) * mc.n_bins), 0, 255), 'uint8'), name='post-process')(out)
        decoder = Model(inputs=[z_in, temperature], outputs=[out])
        return decoder

    def split2d_reverse(self, out, temperature, level_idx):
        layer_key = f'li-{level_idx}'
        split_2d = self.get_layer(Split2d, layer_key,
                                  n_ch=K.int_shape(out)[-1],
                                  bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
        out = split_2d(out, reverse=True, temperature=temperature)
        return out

    def revnet2d(self, out, level_idx, reverse=False):
        for depth_idx in range(self.config.model.n_depth):
            out = self.revnet2d_step(out, level_idx, depth_idx, reverse=reverse)
        return out

    def revnet2d_step(self, out, level_idx, depth_idx, reverse=False):
        layer_key = f'li-{level_idx}/di-{depth_idx:02d}'
        shape = K.int_shape(out)  # tuple of (H, W, C)
        n_ch = shape[-1]
        assert n_ch % 2 == 0, f'n_ch is {n_ch}, shape={shape}'

        act_norm = self.get_layer(ActNorm, layer_key, bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
        inv_1x1_conv = self.get_layer(Invertible1x1Conv, layer_key, bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
        if not reverse:
            out = act_norm(out)
            out = inv_1x1_conv(out)   # implemented only invertible_1x1_conv version
            affine_coupling = self.get_layer(AffineCoupling, layer_key,
                                             n_ch=K.int_shape(out)[-1],
                                             hidden_channel_size=self.config.model.hidden_channel_size,
                                             bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
            out = affine_coupling(out)  # implemented only affine(flow) coupling version
        else:
            affine_coupling = self.get_layer(AffineCoupling, layer_key,
                                             n_ch=K.int_shape(out)[-1],
                                             hidden_channel_size=self.config.model.hidden_channel_size,
                                             bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
            out = affine_coupling(out, reverse=True)
            out = inv_1x1_conv(out, reverse=True)
            out = act_norm(out, reverse=True)
        return out

    def split2d(self, out, level_idx):
        layer_key = f'li-{level_idx}'
        split_2d = self.get_layer(Split2d, layer_key,
                                  n_ch=K.int_shape(out)[-1],
                                  bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
        out = split_2d(out)
        return out

    def get_layer(self, kls, layer_key, **kwargs):
        if (kls, layer_key) not in self._layers:
            self._layers[(kls, layer_key)] = kls(name=f'{kls.__name__}/{layer_key}', **kwargs)
        return self._layers.get((kls, layer_key))


class ActNorm(Layer):
    log_scale = None
    bias = None

    def __init__(self, bit_per_sub_pixel_factor=1, use_loss=True, *args, **kwargs):
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


class Invertible1x1Conv(Layer):
    rotate_matrix = None  # type: tf.Variable

    def __init__(self, bit_per_sub_pixel_factor, *args, **kwargs):
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

        # add log-det as loss
        log_det_factor = int(input_shape[1] * input_shape[2])
        log_det = tf.log(tf.abs(tf.matrix_determinant(self.rotate_matrix)))
        self.add_loss(-1 * log_det_factor * log_det * self.bit_per_sub_pixel_factor)

        # final
        super().build(input_shape)

    def call(self, inputs, reverse=False, **kwargs):
        w = self.rotate_matrix
        if reverse:
            w = tf.matrix_inverse(w)
        w = tf.expand_dims(tf.expand_dims(w, axis=0), axis=0)
        z = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
        return z


class AffineCoupling(Network):  # FlowCoupling
    def __init__(self, n_ch, hidden_channel_size, bit_per_sub_pixel_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_channel_size = hidden_channel_size
        self.bit_per_sub_pixel_factor = bit_per_sub_pixel_factor
        self.conv1 = Conv2D(filters=self.hidden_channel_size,
                            kernel_size=3, strides=1, padding="same", use_bias=False)
        self.actnorm1 = ActNorm(use_loss=False)

        self.conv2 = Conv2D(filters=self.hidden_channel_size,
                            kernel_size=1, strides=1, padding="same", use_bias=False)
        self.actnorm2 = ActNorm(use_loss=False)

        self.last_conv = Conv2D(filters=n_ch, kernel_size=3, padding="same",
                                kernel_initializer='zero', bias_initializer='zero')
        self.outputs = []  # avoid error when __call__()

    def call(self, inputs, reverse=False, **kwargs):
        z = inputs
        z1, z2 = split_channels(z)

        scale, shift = split_channels(self.nn(z1))
        scale = K.exp(scale)  # K.sigmoid(x + 2)  ??
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
    def __init__(self, n_ch, bit_per_sub_pixel_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bit_per_sub_pixel_factor = bit_per_sub_pixel_factor
        self.conv = Conv2D(filters=n_ch, kernel_size=3, padding="same",
                           kernel_initializer='zero', bias_initializer='zero')
        self.outputs = []  # avoid error in __call__()

    def call(self, inputs, reverse=False, temperature=None, **kwargs):
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
            z1 = inputs
            h = self.conv(z1)  # (w, h, n_ch)
            pz = GaussianDiag(h)  # (w, h, n_ch//2)
            z2 = pz.sample2(pz.eps * K.reshape(temperature, [-1, 1, 1, 1]))
            out = K.concatenate([z1, z2], axis=3)
        return out


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

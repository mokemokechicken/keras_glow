from logging import getLogger

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Lambda, Dense, Conv2D
from tensorflow.python.keras import backend as K

from keras_glow.config import Config
import numpy as np

logger = getLogger(__name__)


class GlowModel:
    def __init__(self, config: Config):
        self.config = config
        self._layers = {}

    def build(self):
        mc = self.config.model
        dc = self.config.data
        logger.info(mc)
        in_x = Input(shape=(dc.image_height, dc.image_width, 3), name='image')

        # preprocess
        out = Lambda(lambda x: x / mc.n_bins - 0.5)(in_x)
        # encoder
        out = self.encoder(out)

    def encoder(self, out):
        mc = self.config.model
        for level_idx in range(mc.n_levels):
            out = self.revnet2d(out, level_idx)
            if level_idx < mc.n_levels - 1:
                out = self.split2d(out, level_idx)
        return out

    def revnet2d(self, out, level_idx, reverse=False):
        for depth_idx in range(self.config.model.n_depth):
            out = self.revnet2d_step(out, level_idx, depth_idx, reverse=reverse)
        return out

    def revnet2d_step(self, out, level_idx, depth_idx, reverse=False):
        layer_key = f'li-{level_idx}/di-{depth_idx:02d}'
        shape = K.int_shape(out)  # tuple of (None, H, W, C)
        n_ch = shape[3]
        assert n_ch % 2 == 0

        act_norm = self.get_layer(ActNorm, layer_key)
        inv_1x1_conv = self.get_layer(Invertible1x1Conv, layer_key)
        affine_coupling = self.get_layer(AffineCoupling, layer_key)

        if not reverse:
            out = act_norm(out)
            out = inv_1x1_conv(out)   # implemented only invertible_1x1_conv version
            out = affine_coupling(out)  # implemented only affine(flow) coupling version
        return out

    def split2d(self, out, level_idx):
        return out

    def get_layer(self, kls, layer_key, **kwargs):
        if (kls, layer_key) not in self._layers:
            self._layers[(kls, layer_key)] = kls(name=f'{kls.__name__}/{layer_key}', **kwargs)
        return self._layers.get((kls, layer_key))


class ActNorm(Layer):
    log_scale = None
    bias = None

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
            log_det_factor = input_shape[1]*input_shape[2]

        # DDI(Data-Dependent-Init?) is not implemented
        self.log_scale = self.add_weight('log_scale', shape=var_shape, initializer='zeros')
        self.bias = self.add_weight('bias', shape=var_shape, initializer='zeros')

        # Log-Determinant
        # it seems that this is required only for encoding.
        self.add_loss(-log_det_factor * K.sum(self.log_scale))  # or K.sum(K.abs(self.log_scale)) ???

        # final
        super().build(input_shape)

    def call(self, inputs, reverse=False, **kwargs):
        x = inputs[0]
        if not reverse:
            return (x + self.bias) * K.exp(self.log_scale)
        else:
            return x / K.exp(self.log_scale) - self.bias


class Invertible1x1Conv(Layer):
    rotate_matrix = None  # type: tf.Variable

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        w_shape = [input_shape[3], input_shape[3]]  # [n_channel, n_channel]

        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        self.rotate_matrix = self.add_weight("rotate_matrix", w_shape, initializer=w_init)

        # add log-det as loss
        log_det_factor = input_shape[1] * input_shape[2]
        log_det = tf.log(tf.abs(tf.matrix_determinant(self.rotate_matrix)))
        self.add_loss(-log_det_factor * log_det)

        # final
        super().build(input_shape)

    def call(self, inputs, reverse=False, **kwargs):
        w = self.rotate_matrix
        if reverse:
            w = tf.matrix_inverse(w)
        w = tf.expand_dims(tf.expand_dims(w, axis=0), axis=0)
        z = tf.nn.conv2d(inputs[0], w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
        return z


class AffineCoupling(Layer):  # FlowCoupling
    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        n_ch = input_shape[-1]
        

    def call(self, inputs, **kwargs):
        z = inputs[0]
        shape = K.int_shape(z)
        n_ch = shape[-1]
        z1 = z[:, :, :, :n_ch // 2]
        z2 = z[:, :, :, n_ch // 2:]

        scale_and_shift = self.nn(z1)  # in_ch is n_ch//2, out_ch should be n_ch
        scale = K.exp(scale_and_shift[:, :, :, 0::2])  # K.sigmoid(x + 2)  ??
        shift = scale_and_shift[:, :, :, 1::2]
        z2 = (z2 + shift) * scale
        out = K.concatenate([z1, z2], axis=3)

        self.add_loss(-K.sum(K.log(scale), axis=[1, 2, 3]))
        return out

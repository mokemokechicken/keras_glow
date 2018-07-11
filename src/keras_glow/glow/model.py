from logging import getLogger

from tensorflow.python.keras import Input
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.layers import Lambda, Dense
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
            out = self.revnet2d(level_idx, out)
            if level_idx < mc.n_levels - 1:
                out = self.split2d(level_idx, out)
        return out

    def revnet2d(self, level_idx, out, reverse=False):
        shape = K.int_shape(out)  # tuple of (None, H, W, C)
        n_ch = shape[3]
        assert n_ch % 2 == 0

        act_norm = self.get_layer(ActNorm, level_idx)

        if not reverse:
            out = act_norm(out)

        return out

    def split2d(self, level_idx, out):
        return out

    def get_layer(self, kls, index, **kwargs):
        if (kls, index) not in self._layers:
            self._layers[(kls, index)] = kls(name=f'{kls.__name__}-{index}', **kwargs)
        return self._layers.get((kls, index))


class ActNorm(Layer):
    log_scale = None
    bias = None

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        n_ch = input_shape[-1]
        assert len(input_shape) in (2, 4), f'invalid input_shape={input_shape}'

        var_shape = (1, n_ch) if len(input_shape) == 2 else (1, 1, 1, n_ch)

        # DDI(Data-Dependent-Init?) is not implemented
        self.log_scale = self.add_weight('log_scale', shape=var_shape, initializer='zeros')
        self.bias = self.add_weight('bias', shape=var_shape, initializer='zeros')

        # Log-Determinant
        # it seems that this is required only for encoding.
        self.add_loss(-1 * K.sum(self.log_scale))  # or K.sum(K.abs(self.log_scale)) ???

        # final
        super().build(input_shape)

    def call(self, inputs, reverse=False, **kwargs):
        x = inputs[0]
        if not reverse:
            return (x + self.bias) * K.exp(self.log_scale)
        else:
            return x / K.exp(self.log_scale) - self.bias


class Invertible1x1Conv(Layer):
    rotate_matrix = None

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        # TODO: impl
        w_shape = [input_shape[3], input_shape[3]]  # [n_channel, n_channel]
        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')

        self.rotate_matrix = self.add_weight("rotate_matrix", w_shape, initializer=w_init)

    def call(self, inputs, **kwargs):
        # TODO: impl
        pass

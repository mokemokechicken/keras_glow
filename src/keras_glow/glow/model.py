from logging import getLogger

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import Network
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import load_model
from tensorflow.python.layers.core import Dense

from keras_glow.config import Config
from keras_glow.glow.model_parts import ActNorm, Invertible1x1Conv, AffineCoupling, Squeeze2d, Unsqueeze2d, Split2d, \
    GaussianDiag, PreProcess, AddRandomUniform, PostProcess

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
        if self.encoder is not None:
            logger.info(f"saving encoder to {rc.encoder_path}")
            self.encoder.save(rc.encoder_path)
        if self.decoder is not None:
            logger.info(f"saving decoder to {rc.decoder_path}")
            self.decoder.save(rc.decoder_path, include_optimizer=False)

    def load_all(self):
        rc = self.config.resource
        custom_objects = dict(
            PreProcess=PreProcess,
            PostProcess=PostProcess,
            AddRandomUniform=AddRandomUniform,
            ActNorm=ActNorm,
            Invertible1x1Conv=Invertible1x1Conv,
            AffineCoupling=AffineCoupling,
            Squeeze2d=Squeeze2d,
            Unsqueeze2d=Unsqueeze2d,
            Split2d=Split2d,
        )
        if rc.encoder_path.exists():
            logger.info(f"loading encoder from {rc.encoder_path}")
            self.encoder = load_model(rc.encoder_path, custom_objects=custom_objects, compile=False)
        if rc.decoder_path.exists():
            logger.info(f"loading decoder from {rc.decoder_path}")
            self.decoder = load_model(rc.decoder_path, custom_objects=custom_objects, compile=False)

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
        out = PreProcess(n_bins=mc.n_bins, name="pre-process")(in_x)
        # add noise
        out = AddRandomUniform(n_bins=mc.n_bins, name='add_random_uniform')(out)

        # encoder_loop
        encoder_loop_out = self.build_encoder_loop(out)
        encoder = Model(inputs=in_x, outputs=encoder_loop_out, name="encoder")

        # add prior loss
        prior = GaussianDiag.prior(K.shape(encoder_loop_out))
        encoder.add_loss(-prior.logp(encoder_loop_out) * self.bit_per_sub_pixel_factor)

        # `objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z)[1:])`
        encoder.add_loss(np.log(mc.n_bins) * np.prod(in_shape) * self.bit_per_sub_pixel_factor)
        return encoder

    def build_encoder_loop(self, out):
        mc = self.config.model
        for level_idx in range(mc.n_levels):
            out = Squeeze2d(name=f'Squeeze2d/li-{level_idx}')(out)
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
                out = self.split2d(out, level_idx, reverse=True, temperature=temperature)
            out = self.revnet2d(out, level_idx, reverse=True)
            out = Unsqueeze2d(name=f'Unsqueeze2d/li-{level_idx}')(out)
        # post-process
        out = PostProcess(n_bins=mc.n_bins, name='post-process')(out)
        decoder = Model(inputs=[z_in, temperature], outputs=[out], name='decoder')
        return decoder

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
                                             in_shape=K.int_shape(out),
                                             hidden_channel_size=self.config.model.hidden_channel_size,
                                             bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
            out = affine_coupling(out)  # implemented only affine(flow) coupling version
        else:
            affine_coupling = self.get_layer(AffineCoupling, layer_key,
                                             in_shape=K.int_shape(out),
                                             hidden_channel_size=self.config.model.hidden_channel_size,
                                             bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
            out = affine_coupling(out, reverse=True)
            out = inv_1x1_conv(out, reverse=True)
            out = act_norm(out, reverse=True)
        return out

    def split2d(self, out, level_idx, reverse=False, temperature=None):
        layer_key = f'li-{level_idx}'
        split_2d = self.get_layer(Split2d, layer_key,
                                  in_shape=K.int_shape(out),
                                  bit_per_sub_pixel_factor=self.bit_per_sub_pixel_factor)
        if not reverse:
            out = split_2d(out, reverse=reverse)
        else:
            out = split_2d([out, temperature], reverse=reverse)
        return out

    def get_layer(self, kls, layer_key, **kwargs):
        if (kls, layer_key) not in self._layers:
            self._layers[(kls, layer_key)] = kls(name=f'{kls.__name__}/{layer_key}', **kwargs)
        return self._layers.get((kls, layer_key))

    @property
    def decoder_input_shape(self):
        if self.decoder is not None:
            return K.int_shape(self.decoder.input[0])[1:]
        return None

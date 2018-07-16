import os
from logging import getLogger

from keras_glow.config import Config
from keras_glow.glow.model import GlowModel
import numpy as np
from PIL import Image

from keras_glow.lib.file_util import create_basedir

logger = getLogger(__name__)


class Agent:
    def __init__(self, config: Config, model: GlowModel):
        self.config = config
        self.model = model

    def sample(self, n=None, temperature=None):
        n = n or 1
        temperature = 0.7 if temperature is None else temperature
        logger.info(f"sampling n={n} temperature={temperature}")
        shape = self.model.decoder_input_shape
        z = np.random.normal(loc=0, scale=1, size=(n, ) + shape)
        t = np.ones((n, )) * temperature
        images = self.model.decoder.predict([z, t])
        return images

    def sample_to_save(self, n=None, temperature=None, image_path_base=None):
        image_path_base = image_path_base or self.config.resource.sample_image_base
        images = self.sample(n=n, temperature=temperature)

        create_basedir(image_path_base)
        for i, img_array in enumerate(images):
            img = Image.fromarray(img_array)
            img.save(f'{image_path_base}_{i:03d}.png')

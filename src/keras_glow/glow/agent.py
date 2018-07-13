from logging import getLogger

from keras_glow.config import Config
from keras_glow.glow.model import GlowModel
import numpy as np

logger = getLogger(__name__)


class Agent:
    def __init__(self, config: Config, model: GlowModel):
        self.config = config
        self.model = model

    def sample(self, n=1, temperature=0.7):
        shape = self.model.decoder_input_shape
        z = np.random.normal(loc=0, scale=1, size=(n, ) + shape)
        t = np.ones((n, )) * temperature
        images = self.model.decoder.predict([z, t])
        return images


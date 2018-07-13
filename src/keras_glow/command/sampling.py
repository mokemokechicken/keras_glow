from logging import getLogger

from PIL import Image

from keras_glow.config import Config
from keras_glow.data.data_processor import DataProcessor
from keras_glow.glow.agent import Agent
from keras_glow.glow.model import GlowModel
from keras_glow.glow.trainer import Trainer

logger = getLogger(__name__)


def start(config: Config):
    logger.debug("start")
    SamplingCommand(config).start()


class SamplingCommand:
    def __init__(self, config: Config):
        self.config = config

    def start(self):
        model = GlowModel(self.config)
        model.load_all()

        agent = Agent(self.config, model)
        images = agent.sample(n=1)
        logger.debug(images[0].shape)

        image_base = self.config.resource.sample_image_base

        for i, img_array in enumerate(images):
            img = Image.fromarray(img_array)
            img.save(f'{image_base}_{i:03d}.png')
            img.close()

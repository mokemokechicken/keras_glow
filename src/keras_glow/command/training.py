from logging import getLogger

from keras_glow.config import Config
from keras_glow.data.training.processor import Processor
from keras_glow.glow.model import GlowModel

logger = getLogger(__name__)


def start(config: Config):
    logger.info("start")
    TrainingCommand(config).start()


class TrainingCommand:
    def __init__(self, config: Config):
        self.config = config

    def start(self):
        dp = Processor(self.config)
        dataset = dp.create_dataset()

        model = GlowModel(self.config)
        model.build()



import json
from logging import getLogger

from keras_glow.config import Config
from keras_glow.data.data_processor import DataProcessor
from keras_glow.glow.model import GlowModel
from keras_glow.glow.trainer import Trainer

logger = getLogger(__name__)


def start(config: Config):
    logger.debug("start")
    TrainingCommand(config).start()


class TrainingCommand:
    def __init__(self, config: Config):
        self.config = config

    def start(self):
        dp = DataProcessor(self.config)

        model = GlowModel(self.config)
        if self.config.runtime.args.new or not self.config.resource.encoder_path.exists():
            logger.info("create a new model")
            model.build()
        else:
            logger.info("loading a existing model")
            model.load_all()

        trainer = Trainer(self.config)
        trainer.fit(model, dp)

        model.save_all()

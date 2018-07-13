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
        # for y in dp.iterator():
        #     print(y.shape)

        model = GlowModel(self.config)
        model.build()

        # with open("enc.json", "wt") as f:
        #     f.write(json.dumps(model.encoder.get_config(), indent=2))
        # with open("dec.json", "wt") as f:
        #     f.write(json.dumps(model.decoder.get_config(), indent=2))

        trainer = Trainer(self.config)
        trainer.fit(model, dp)

        model.save_all()

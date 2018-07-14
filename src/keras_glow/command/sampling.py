from logging import getLogger

from keras_glow.config import Config
from keras_glow.glow.agent import Agent
from keras_glow.glow.model import GlowModel

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

        model.encoder.get_layer()

        #agent = Agent(self.config, model)
        #agent.sample_to_save(n=10)

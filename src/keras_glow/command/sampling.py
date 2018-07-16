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
        args = self.config.runtime.args
        assert int(args.n) > 0 and 0 <= args.t <= 1
        model = GlowModel(self.config)
        model.load_all()

        agent = Agent(self.config, model)
        agent.sample_to_save(n=int(args.n), temperature=args.t)

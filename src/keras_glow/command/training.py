from logging import getLogger

from keras_glow.config import Config


logger = getLogger(__name__)


def start(config: Config):
    logger.info("start")

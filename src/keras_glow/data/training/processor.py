from keras_glow.config import Config
from keras_glow.data.training.dataset import Dataset


class Processor:
    def __init__(self, config: Config):
        self.config = config

    def create_dataset(self):
        return Dataset()

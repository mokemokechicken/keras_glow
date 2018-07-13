import os
import re
from pathlib import Path

from moke_config import ConfigBase


def _system_dir():
    return Path(__file__).parent.parent.parent


def _data_dir():
    return _system_dir() / 'data'


class Config(ConfigBase):
    def __init__(self):
        self.runtime = RuntimeConfig()
        self.resource = ResourceConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()


class RuntimeConfig(ConfigBase):
    def __init__(self):
        self.args = None


class ResourceConfig(ConfigBase):
    def __init__(self):
        self.system_dir = _system_dir()

        # data
        self.data_dir = self.system_dir / 'data'
        self.image_dir = self.data_dir / 'image'
        self.model_dir = self.data_dir / 'model'
        self.sample_dir = self.data_dir / 'sample'

        # Model
        self.encoder_path = self.model_dir / 'encoder.h5'
        self.decoder_path = self.model_dir / 'decoder.h5'

        # Sample
        self.sample_image_base = self.sample_dir / 'img'

        # Log
        self.log_dir = self.system_dir / "log"
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.main_log_path = self.log_dir / "main.log"

    def create_base_dirs(self):
        dirs = [self.log_dir, self.model_dir, self.sample_dir]

        for d in dirs:
            os.makedirs(d, exist_ok=True)


class DataConfig(ConfigBase):
    def __init__(self):
        self.image_width = 32
        self.image_height = 32


class ModelConfig(ConfigBase):
    def __init__(self):
        self.n_bins = 256
        self.n_levels = 2  # 4
        self.n_depth = 1   # 32
        self.hidden_channel_size = 16  # 512


class TrainingConfig(ConfigBase):
    def __init__(self):
        self.batch_size = 1
        self.lr = 0.0001
        self.lr_patience = 5
        self.lr_decay = 0.1
        self.epochs = 10
        self.steps_per_epoch = 1  # None means auto calculated
        self.sample_every_n_epoch = 5
        self.sample_n_image = 10

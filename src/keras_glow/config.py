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


class RuntimeConfig(ConfigBase):
    def __init__(self):
        self.args = None


class ResourceConfig(ConfigBase):
    def __init__(self):
        self.system_dir = _system_dir()

        # Log
        self.log_dir = self.system_dir / "log"
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.main_log_path = self.log_dir / "main.log"

    def create_base_dirs(self):
        dirs = [self.log_dir]

        for d in dirs:
            os.makedirs(d, exist_ok=True)


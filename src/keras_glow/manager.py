import argparse
from logging import getLogger
import importlib

from moke_config import create_config

from keras_glow.lib.file_util import load_yaml_from_file
from .config import Config
from .lib.logger import setup_logger

logger = getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    def add_common_options(p):
        p.add_argument("--config", help="specify config file")
        p.add_argument("--log-level", help="specify Log Level(debug/info/warning/error): default=info",
                       choices=['debug', 'info', 'warning', 'error'])

    sub_parser = sub.add_parser("training")
    sub_parser.set_defaults(command='training')
    sub_parser.add_argument('--new', action="store_true", help='start training new model')
    add_common_options(sub_parser)

    sub_parser = sub.add_parser("sampling")
    sub_parser.add_argument('-n', type=int, help='number of sampling(default=10)', default=10)
    sub_parser.add_argument('-t', type=float, help='temperature of sampling(default=0.7)', default=0.7)
    sub_parser.set_defaults(command='sampling')
    add_common_options(sub_parser)
    return parser


def setup(config: Config, args):
    config.resource.create_base_dirs()
    setup_logger(config.resource.main_log_path, level=args.log_level or 'info')
    config.runtime.args = args


def start():
    parser = create_parser()
    args = parser.parse_args()
    if args.config:
        config_dict = load_yaml_from_file(args.config)
    else:
        config_dict = {}
    config = create_config(Config, config_dict)  # type: Config
    setup(config, args)
    logger.info(args)

    if hasattr(args, "command"):
        m = importlib.import_module(f'keras_glow.command.{args.command}')
        m.start(config)
    else:
        parser.print_help()
        raise RuntimeError(f"unknown command")

    logger.debug(f"Finish: {args}")

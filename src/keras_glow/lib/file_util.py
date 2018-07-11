import gzip
import os

import yaml


def create_basedir(file_path):
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def open_file(filename, mode='rt'):
    if filename.endswith('.gz'):
        in_file = gzip.open(filename, mode)
    else:
        in_file = open(filename, mode)
    return in_file


def write_yaml_to_file(filepath, data):
    with open(filepath, "wt") as f_out:
        try:
            yaml.dump(data, f_out, default_flow_style=False)
        except TypeError:
            yaml.dump(eval(str(data)), f_out, default_flow_style=False)


def load_yaml_from_file(filepath):
    with open(filepath, "rt") as f_in:
        return yaml.load(f_in)

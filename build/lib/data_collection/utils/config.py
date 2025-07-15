"""
    Utils for Loading Configs
"""


"""
    Standard Libraries
"""
from typing import Dict

"""
    3rd Party Libraries
"""
import yaml


def load_config(config_path: str) -> Dict:
    """
        Load YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


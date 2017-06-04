"""
Library with common functions.
"""

import logging
import os

HARVEST_LOG = 'var/log/harvest.log'
DATA = 'var/data'


def mkdirs(path):
    """Create a directory for a path if it doesn't exist yet."""
    dirname = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass


def setup_logging(name, filename):
    mkdirs(filename)
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

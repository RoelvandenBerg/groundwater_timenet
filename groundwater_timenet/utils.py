"""
Library with common functions.
"""

import os


def mkdirs(path):
    """Create a directory for a path if it doesn't exist yet."""
    dirname = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass
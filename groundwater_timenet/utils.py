"""
Library with common functions.
"""

import logging
import os

from osgeo import ogr
from osgeo import osr


PARSE_LOG = 'var/log/parse.log'
HARVEST_LOG = 'var/log/harvest.log'
DATA = 'var/data'


def mkdirs(path):
    """Create a directory for a path if it doesn't exist yet."""
    dirname = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass


def setup_logging(name, filename, loglevel=logging.DEBUG):
    mkdirs(filename)
    logging.basicConfig(filename=filename, level=loglevel)
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def transform(geom, source_epsg=4326, target_epsg=28992):
    source = osr.SpatialReference()
    source.ImportFromEPSG(source_epsg)
    target = osr.SpatialReference()
    target.ImportFromEPSG(target_epsg)
    transformation = osr.CoordinateTransformation(source, target)
    geom.Transform(transformation)


def point(x, y):
    p = ogr.Geometry(ogr.wkbPoint)
    p.AddPoint(x, y)
    return p


def multipoint(points):
    mp = ogr.Geometry(ogr.wkbMultiPoint)
    for x, y in points:
        mp.AddGeometry(point(x, y))
    return mp


def closest_point(point, multipoint):
    return sorted(
        [(mp.Distance(point), i) for i, mp in enumerate(multipoint)])[0][1]


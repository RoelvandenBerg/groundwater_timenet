"""
Library with common functions.
"""

import logging
import os
from math import ceil

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


def within(geom, minx, miny, maxx, maxy):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minx, miny)
    ring.AddPoint(maxx, miny)
    ring.AddPoint(maxx, maxy)
    ring.AddPoint(minx, maxy)
    ring.AddPoint(minx, miny)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    # TODO: Fix this, within seems to be within the envelope, which is
    # useless for our case since the grid is based on the envelope.
    return poly.Within(geom) or poly.Intersects(geom)


def sliding_geom_window(source_json, gridHeight=10000, gridWidth=10000):
    """
    Generates bounding boxes that fit a shape.
    Leans heavily on https://pcjericks.github.io/py-gdalogr-cookbook/

    :param source_json: WGS84 geojson of the area the sliding window is for
    :param gridHeight: height in meters of each sliding window grid cell
    :param gridWidth: width in meters of each sliding window grid cell
    :return: grid cell generator with bounding box (minx, miny, maxx, maxy) for
        source json
    """
    driver = ogr.GetDriverByName('GeoJSON')
    data_source = driver.Open(source_json, 0)
    if data_source is None:
        raise ValueError('%s is not a valid json', source_json)
    layer = data_source.GetLayer()
    feature = next(layer)
    geom = feature.geometry()
    # reproject it to Amersfoort / RD New
    transform(geom)
    (xmin, xmax, ymin, ymax) = (round(x) for x in geom.GetEnvelope())

    # get rows
    rows = ceil((ymax-ymin)/gridHeight)
    # get columns
    cols = ceil((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create grid cells
    for _ in range(cols):
        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom = ringYbottomOrigin

        for _ in range(rows):
            if (geom, ringXleftOrigin, ringYtop, ringXrightOrigin, ringYbottom
                    ):
                yield (
                    ringXleftOrigin + gridWidth,
                    ringYtop + gridHeight,
                    ringXrightOrigin + gridWidth,
                    ringYbottom + gridHeight
                )

            # new envelope for next poly
            ringYtop -= gridHeight
            ringYbottom -= gridHeight

        # new envelope for next poly
        ringXleftOrigin += gridWidth
        ringXrightOrigin += gridWidth


def parse_filepath(minx, miny, filename_base="dino"):
    filepath = os.path.join(
        DATA, filename_base, str(miny), str(minx) + ".hdf5"
    )
    mkdirs(filepath)
    return filepath

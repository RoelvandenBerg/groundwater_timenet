"""
Library with common functions.
"""

import logging
import os
from math import ceil

import numpy as np
import h5py
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


def create_sliding_geom_window(
        source_json='NederlandRegion.json', gridHeight=10000, gridWidth=10000):
    """
    Generates bounding boxes that fit a shape.
    Leans heavily on https://pcjericks.github.io/py-gdalogr-cookbook/

    :param source_json: WGS84 geojson of the area the sliding window is for
    :param gridHeight: height in meters of each sliding window grid cell
    :param gridWidth: width in meters of each sliding window grid cell
    :return: grid cell generator with bounding box (minx, miny, maxx, maxy) for
        source json
    """
    source_json = 'NederlandRegion.json'
    gridHeight=10000
    gridWidth=10000
    driver = ogr.GetDriverByName('GeoJSON')
    data_source = driver.Open(source_json, 0)
    if data_source is None:
        raise ValueError('%s is not a valid json', source_json)
    layer = data_source.GetLayer()
    feature = next(layer)
    geom = feature.geometry()
    # reproject it to Amersfoort / RD New
    transform(geom)
    (xmin, xmax, ymin, ymax) = geom.GetEnvelope()
    # make sure to remain within the geometry to prevent segmentation faults:
    xmin, xmax, ymin, ymax = ceil(xmin), int(xmax), ceil(ymin), int(ymax)
    xmin_rounded = int(xmin/gridWidth)*gridWidth
    xmax_rounded = int(ceil(xmax/gridWidth))*gridWidth
    ymin_rounded = int(ymin/gridHeight)*gridHeight
    ymax_rounded = int(ceil(ymax/gridHeight))*gridHeight

    return np.array([
        (xmin_c, ymin_c, xmax_c, ymax_c)
        for xmin_c, ymin_c, xmax_c, ymax_c in
        (
            (
                max(x, xmin),
                max(y - gridHeight, ymin),
                min(x + gridWidth, xmax),
                min(y, ymax)
            ) for y in range(ymax_rounded, ymin_rounded, -gridWidth)
            for x in range(xmin_rounded, xmax_rounded, gridWidth)
        ) if within(geom, xmin_c, ymin_c, xmax_c, ymax_c)
    ])


def sliding_geom_window(
        source_json='NederlandRegion.json', gridHeight=10000, gridWidth=10000,
        source_netcdf="var/data/cache/sliding_geom.nc"):
    geom_array = cache_nc(
        create_sliding_geom_window,  source_netcdf, "geom_window",
        source_json=source_json,
        gridHeight=gridHeight,
        gridWidth=gridWidth
    )
    return (
        (float(a), float(b), float(c), float(d)) for a, b, c, d in
        iter(geom_array)
    )


def store_nc(data, dataset_name, target_nc="var/data/cache/cache.nc"):
    with h5py.File(target_nc, "w", libver='latest') as h5_file:
        dataset = h5_file.create_dataset(
            dataset_name,
            data.shape,
            dtype=data.dtype)
        dataset[...] = data


def cache_nc(source_data_function, target_nc, dataset_name=None,
             decode=None, *source_data_function_args,
             **source_data_function_kwargs
             ):
    dataset_name = dataset_name or os.path.basename(target_nc).strip('.nc')
    if not os.path.exists(target_nc):
        mkdirs(target_nc)
        data = source_data_function(
            *source_data_function_args,
             **source_data_function_kwargs
        )
        store_nc(data, dataset_name, target_nc)
    with h5py.File(target_nc, "r", libver='latest') as target:
        return target[dataset_name][()]


def parse_filepath(minx, miny, filename_base="dino"):
    filepath = os.path.join(
        DATA, filename_base, str(miny), str(minx) + ".hdf5"
    )
    mkdirs(filepath)
    return filepath

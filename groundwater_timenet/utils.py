"""
Library with common functions.
"""

import logging
import os
from math import ceil

from osgeo import ogr
from osgeo import osr
import h5py
import numpy as np


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


def setup_logging(name, filename, level="DEBUG"):
    mkdirs(filename)
    logging.basicConfig(
        filename=filename,
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)
    return logger


logger = setup_logging(__name__, PARSE_LOG, level="DEBUG")


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


def closest_point(x, y, multipoint):
    pt = point(x, y)
    return sorted(
        [(mp.Distance(pt), i) for i, mp in enumerate(multipoint)])[0][1]


def bbox2polygon(minx, miny, maxx, maxy):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minx, miny)
    ring.AddPoint(maxx, miny)
    ring.AddPoint(maxx, maxy)
    ring.AddPoint(minx, maxy)
    ring.AddPoint(minx, miny)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def within(geom, minx, miny, maxx, maxy):
    poly = bbox2polygon(minx, miny, maxx, maxy)
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
    gridHeight = 10000
    gridWidth = 10000
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
        source_netcdf="var/data/cache/sliding_geom.h5"):
    geom_array = cache_h5(
        create_sliding_geom_window,  source_netcdf, "geom_window",
        source_json=source_json,
        gridHeight=gridHeight,
        gridWidth=gridWidth
    )
    return (
        (float(a), float(b), float(c), float(d)) for a, b, c, d in
        iter(geom_array)
    )


def store_h5(
        data, dataset_name, target_h5="var/data/cache/cache.h5", many=False):
    if not many:
        data = [data]
        dataset_name = [dataset_name]
    mkdirs(target_h5)
    with h5py.File(target_h5, "w", libver='latest') as h5_file:
        for i, dataset_data in enumerate(data):
            dataset = h5_file.create_dataset(
                dataset_name[i],
                dataset_data.shape,
                dtype=dataset_data.dtype)
            dataset[...] = dataset_data


def read_h5(filepath, dataset_name, index=None, many=False):
    with h5py.File(filepath, 'r', libver='latest') as h5file:
        if not many:
            if index is None:
                index = ()
            return h5file.get(dataset_name)[index]
        else:
            if index is None:
                index = [() for _ in range(len(dataset_name))]
            return tuple(
                h5file.get(name)[index[i]] for i, name in
                enumerate(dataset_name)
            )


def cache_h5(source_data_function, target_h5, cache_dataset_name=None,
             decode=None, *source_data_function_args,
             **source_data_function_kwargs
             ):
    cache_dataset_name = cache_dataset_name or os.path.basename(
        target_h5).strip('.h5')
    if not os.path.exists(target_h5):
        mkdirs(target_h5)
        data = source_data_function(
            *source_data_function_args,
            **source_data_function_kwargs
        )
        store_h5(data, cache_dataset_name, target_h5)
    with h5py.File(target_h5, "r", libver='latest') as target:
        return target[cache_dataset_name][()]


def parse_filepath(minx, miny, filename_base="dino"):
    filepath = os.path.join(
        DATA, filename_base, str(miny), str(minx) + ".hdf5"
    )
    mkdirs(filepath)
    return filepath


def int_or_nan(x):
    try:
        return int(x)
    except ValueError:
        return np.nan


def try_h5(fn, d=None):
    try:
        f = h5py.File(fn, 'r')
        if d is not None:
            f.get(d)[:]
    except (OSError, TypeError):
        return fn


def tryfloat(f):
    try:
        return float(f)
    except (ValueError, TypeError):
        pass
    return 0.0


def _get_raster_filenames(rootdir, raise_errors, dataset_name=None):
    files = sorted(
        [
            os.path.join(r, f[0]) for r, d, f in os.walk('var/data/' + rootdir)
            if f
        ]
    )
    faulty = [x for x in [try_h5(f, dataset_name) for f in files] if x]
    if faulty:
        message = 'Broken HDF5 files found:\n- {}'.format('\n- '.join(faulty))
        if raise_errors:
            raise OSError(message)
        else:
            logger.debug(message)
            return np.array(
                [f.encode('utf8') for f in files if f not in faulty])
    return np.array([f.encode('utf8') for f in files])


def raster_filenames(
        root, source_netcdf=None, raise_errors=True, dataset_name=None):
    source_netcdf = source_netcdf or "var/data/cache/{}files.h5".format(
        root if raise_errors else "filtered_" + root)
    filenames = cache_h5(
        _get_raster_filenames, source_netcdf,
        cache_dataset_name=root,
        rootdir=root,
        raise_errors=raise_errors,
        dataset_name=dataset_name
    )
    return [f.decode('utf8') for f in filenames]

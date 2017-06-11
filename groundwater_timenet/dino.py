from math import ceil
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.error import HTTPError
from itertools import chain
import logging
import os

from osgeo import ogr
from osgeo import osr
from owslib.wfs import WebFeatureService
from suds.client import Client as SoapClient
import h5py
import numpy as np

try:
    from groundwater_timenet import utils
except ImportError:
    import utils


logger = utils.setup_logging(__name__, utils.HARVEST_LOG, logging.INFO)


WFS_URL = 'http://www.broinspireservices.nl/wfs/osgegmw-a-v1.0'
WFS_LAYER_NAME = 'gdn:Grondwateronderzoek'
SOAP_CLIENT = SoapClient("http://www.dinoservices.nl/gwservices/gws-v11?wsdl")
FILENAME_BASE = "dino"
NAN_VALUE = -9999999


def transform(geom, source_epsg=4326, target_epsg=28992):
    source = osr.SpatialReference()
    source.ImportFromEPSG(source_epsg)
    target = osr.SpatialReference()
    target.ImportFromEPSG(target_epsg)
    transformation = osr.CoordinateTransformation(source, target)
    geom.Transform(transformation)


def get_features(wfs, layer_name, minx, miny, maxx, maxy):
    """
    Generator that iterates over layer features for a certain bounding box.

    :param layer: osr layer
    :param minx: bounding box min x coordinate
    :param miny: bounding box min y coordinate
    :param maxx: bounding box max x coordinate
    :param maxy: bounding box max y coordinate
    :return: feature generator with a tuple with the following attributes for
      each feature:
        ('dino_nr',
         'x_rd_crd',
         'y_rd_crd',
         'top_depth_mv',
         'bottom_depth_mv',
         'top_height_nap',
         'bottom_height_nap',
         'Grondwaterstand|start_date',
         'Grondwaterstand|end_date')
    """
    logger.info("Bounding Box: %d %d %d %d", minx, miny, maxx, maxy)
    resp = wfs.getfeature(typename=layer_name,
                          bbox=(minx, miny, maxx, maxy))
    with NamedTemporaryFile('w') as temporary_file:
        with open(temporary_file.name, 'w') as gmlfile:
            gmlfile.write(resp.read())
        driver = ogr.GetDriverByName('GML')
        gml = driver.Open(temporary_file.name)
        layer = gml.GetLayer()
        if layer is not None:
            logger.debug("Got %d features.", layer.GetFeatureCount())
            for feature in layer:
                yield (
                    try_get_field(feature, 'dino_nr', 1)[0],
                    try_get_field(feature, 'x_rd_crd', 1)[0],
                    try_get_field(feature, 'y_rd_crd', 1)[0],
                    try_get_field(feature, 'Grondwaterstand|start_date', 1)[0],
                    try_get_field(feature, 'Grondwaterstand|end_date', 1)[0],
                    try_get_field(feature, 'top_depth_mv', 2),
                    try_get_field(feature, 'bottom_depth_mv', 2),
                    try_get_field(feature, 'top_height_nap', 2),
                    try_get_field(feature, 'bottom_height_mv', 2),
                )


def try_get_field(feature, fieldname, n, default=""):
    try:
        value = feature.GetField(fieldname)
    except ValueError:
        return [default] * n
    if isinstance(value, list):
        if len(value) >= n:
            return value[:n]
        return value + [default] * (n - len(value))
    elif value is None:
        return [default] * n
    return [value] + [default] * (n - 1)


def load_station_data(nitg_nr):
    periods = [(1900, 2017)]
    meetreeksen = []
    while len(periods) > 0:
        start_year, end_year = periods.pop()
        try:
            meetreeksen = meetreeksen + list(
                SOAP_CLIENT.service.findMeetreeks(
                    WELL_NITG_NR=nitg_nr,
                    START_DATE=str(start_year) + '-01-01',
                    END_DATE=str(end_year) + '-12-01',
                    UNIT='SFL'
                )
            )
        except:
            logger.exception(
                "Suds client is unavailable for well %s, waiting for 10 "
                "minutes, breaking query period in two", nitg_nr)
            sleep(600)
            halfway = int(start_year + (end_year - start_year) / 2)
            periods.append((start_year, halfway))
            periods.append((halfway, end_year))
    return (
        (
            meetreeks.WELL_NITG_NR, meetreeks.WELL_TUBE_NR,
            list(
                (level.DATE, level.LEVEL, level.REMARK)
                for level in meetreeks.LEVELS
            )
        ) for meetreeks in meetreeksen
    )


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
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # TODO: Fix this, within seems to be within the envelope, which is
            # useless for our case since the grid is based on the envelope.
            if poly.Within(geom) or poly.Intersects(geom):
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


def load_dino_grid_cell(features):
    for (well, x, y, start, end,
         (top_depth_mv_up, top_depth_mv_down),
         (bottom_depth_mv_up, bottom_depth_mv_down),
         (top_height_up, top_height_down),
         (bottom_height_up, bottom_height_down)) in features:
        try:
            data = load_station_data(well)
            for well_nr, tube_nr, well_data in data:
                yield ([well, tube_nr, x, y, start, end, top_depth_mv_up,
                        top_depth_mv_down, bottom_depth_mv_up,
                        bottom_depth_mv_down, top_height_up, top_height_down,
                        bottom_height_up, bottom_height_down], well_data)
        except AttributeError:
            logger.info("Well %s doesn't contain values", well)


def load_dino_groundwater(skip=0, url=WFS_URL, layer_name=WFS_LAYER_NAME):
    wfs = WebFeatureService(url=url, version='2.0.0')
    sliding_window = sliding_geom_window('NederlandRegion.json')
    [next(sliding_window) for _ in range(skip)]
    for minx, miny, maxx, maxy in sliding_window:
        features = get_features(wfs, layer_name, minx, miny, maxx, maxy)
        yield load_dino_grid_cell(features), minx, miny


def parse_filepath(minx, miny, filename_base="dino"):
    filepath = os.path.join(
        utils.DATA, filename_base, str(miny), str(minx) + ".hdf5"
    )
    utils.mkdirs(filepath)
    return filepath


def download_hdf5(skip=0, filename_base=FILENAME_BASE):
    dino_data = load_dino_groundwater(skip)
    skip_filepath = os.path.join(utils.DATA, filename_base, "skip_count.txt")
    total_count = 0
    for grid_cell, minx, miny, in dino_data:
        filepath = parse_filepath(minx, miny, filename_base)
        h5_file = h5py.File(filepath, "w", libver='latest')
        meta_data = []
        changed = False
        for metadata, well_data in grid_cell:
            # cast to float and handle faulty data.
            data_ = [
                [np.datetime64(d, 's').astype('f4'), v]
                for d, v, f in well_data if f is None and v is not None
            ]
            if len(data_) == 0:
                logger.info("Well %s %s doesn't contain data",
                             metadata[0], metadata[1])
                continue
            data = np.array(data_)
            logger.info(
                "Got Feature: %s, size: %s", str(metadata), data.shape[0]
            )
            try:
                dataset = h5_file.create_dataset(
                    metadata[0] + str(metadata[1]),
                    data.shape,
                    maxshape=(None, 2),
                    dtype='f4')
            except RuntimeError:
                del h5_file[metadata[0] + str(metadata[1])]
                dataset = h5_file.create_dataset(
                    metadata[0] + str(metadata[1]),
                    data.shape,
                    maxshape=(None, 2),
                    dtype='f4')
                logger.warn("%s %s ALREADY EXISTS! Deleted.",
                             metadata[0], str(metadata[1]))
            if dataset.shape != data.shape:
                dataset.resize(data.shape)
                logger.warn("%s %s HAS WRONG SHAPE! Resized.",
                         metadata[0], str(metadata[1]))
            dataset[...] = data
            meta_data.append([str(x) for x in metadata])
            changed = True
        if changed:
            meta_data_array = np.array([[u.encode('utf8') for u in record]
                                        for record in meta_data])
            try:
                meta_dataset = h5_file.create_dataset(
                    "metadata", meta_data_array.shape,
                    dtype=str(meta_data_array.dtype)
                )
            except RuntimeError:
                meta_dataset = h5_file.get("metadata")
            meta_dataset[...] = meta_data_array
            count = len(meta_data)
            total_count += count
            logger.info(
                'Downloaded %d wells to %s. Total count: %d, next time, skip %d '
                'valid grid cells.', count, filepath, total_count, skip
            )
        if not changed:
            os.remove(filepath)
        skip += 1
        with open(skip_filepath, 'w') as skip_file:
            skip_file.write(str(skip))


def list_metadata():
    base = os.path.join(utils.DATA, FILENAME_BASE)
    logger.info("All y coordinates: %s", str(os.listdir(base)))
    total = 0
    for root, dirs, files in os.walk(base):
        hdf5_files = [
            os.path.join(root, hdf5_file) for hdf5_file in files
            if hdf5_file[-4:] == "hdf5"
        ]
        for filepath in hdf5_files:
            h5_file = h5py.File(filepath, "r")
            length = len(h5_file.get("metadata", []))
            total += length
            if length == 0:
                message = "File " + filepath + "doesn't contain metadata"
            else:
                message = "File " + filepath + "contains " + str(length) + \
                          " records"
            logger.info(message)
    logger.info("Total records found: %d", total)


if __name__ == "__main__":
    list_metadata()
    skip_filepath = os.path.join(utils.DATA, FILENAME_BASE, "skip_count.txt")
    utils.mkdirs(skip_filepath)
    try:
        with open(skip_filepath, 'r') as skip_file:
            skip = int(skip_file.read())
    except OSError:
        skip = 0
    download_hdf5(skip, FILENAME_BASE)

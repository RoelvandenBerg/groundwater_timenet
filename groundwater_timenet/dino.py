import logging
import sys
from math import ceil

from suds.client import Client as SoapClient
from osgeo import ogr
from osgeo import osr
from osgeo import gdal

logger = logging.getLogger(__name__)

WFS_URL = 'http://www.broinspireservices.nl/wfs/osgegmw-a-v1.0'
WFS_LAYER_NAME = 'Grondwateronderzoek'

SOAP_CLIENT = SoapClient("http://www.dinoservices.nl/gwservices/gws-v11?wsdl")


# Speeds up querying WFS capabilities for services with alot of layers
gdal.SetConfigOption('OGR_WFS_LOAD_MULTIPLE_LAYER_DEFN', 'NO')

# Set config for paging. Works on WFS 2.0 services and WFS 1.0 and 1.1 with
# some other services.
gdal.SetConfigOption('OGR_WFS_PAGING_ALLOWED', 'YES')
gdal.SetConfigOption('OGR_WFS_PAGE_SIZE', '10000')


def wfs_layer_and_srs(url=WFS_URL, layer_name=WFS_LAYER_NAME):
    driver = ogr.GetDriverByName('WFS')
    wfs = driver.Open('WFS:' + url)
    if not wfs:
        sys.exit('ERROR: can not open WFS datasource')
    else:
        pass
    layer = wfs.GetLayerByName(layer_name)
    if not layer:
        sys.exit('ERROR: can not open WFS layer')
    else:
        pass
    srs = layer.GetSpatialRef()
    logger.info(
        'Using Dino WFS serice "%s", with layer "%s" and spatial reference: %s',
        url,
        layer_name,
        srs.GetAttrValue('projcs')
    )
    return layer, srs


def transform(geom, source_epsg=4326, target_epsg=28992):
    source = osr.SpatialReference()
    source.ImportFromEPSG(2927)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    transformation = osr.CoordinateTransformation(source, target)
    geom.Transform(transformation)


def get_features(layer, minx, miny, maxx, maxy):
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
    layer.SetSpatialFilterRect(minx, miny, maxx, maxy)
    for i in layer.GetFeatureCount():
        feature = layer.GetFeature(i)
        yield (
            feature.GetField('dino_nr'),
            feature.GetField('x_rd_crd'),
            feature.GetField('y_rd_crd'),
            feature.GetField('Grondwaterstand|start_date'),
            feature.GetField('Grondwaterstand|end_date'),
            feature.GetField('top_depth_mv'),
            feature.GetField('bottom_depth_mv'),
            feature.GetField('top_height_nap'),
            feature.GetField('bottom_depth_mv'),
        )


def load_station_data(nitg_nr):
    meetreeksen = SOAP_CLIENT.service.findMeetreeks(
        WELL_NITG_NR=nitg_nr,
        START_DATE='1900-01-01',
        END_DATE='2017-12-01',
        UNIT='SFL'
    )
    return (
        (
            meetreeks.WELL_NITG_NR, meetreeks.WELL_TUBE_NR,
            (
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
        raise ValueError('%s is not a valid shapefile', source_json)
    layer = data_source.GetLayer()
    feature = next(layer)
    geom = feature.geometry()
    # reproject it to Amersfoort / RD New
    transform(geom)
    (xmin, xmax, ymin, ymax) = geom.GetEnvelope()

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
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1

            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # TODO: test this, within might be within the envelope, which is
            # useless for our case since the grid is based on the envelope.
            if poly.Within(geom):
                yield (
                    ringXleftOrigin,
                    ringYtopOrigin,
                    ringXrightOrigin,
                    ringYbottomOrigin
                )

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth


def load_dino_groundwater(url=WFS_URL, layer_name=WFS_LAYER_NAME):
    layer, srs = wfs_layer_and_srs(url, layer_name)
    sliding_window = sliding_geom_window('NederlandRegion.json')
    for minx, miny, maxx, maxy in sliding_window:
        features = get_features(layer, minx, miny, maxx, maxy)
        for (well, x, y, top_depth, bottom_depth, top_height, bottom_height,
                start, end) in features:
            data = load_station_data(well)
            for i, (well_nr, tube_nr, well_data) in enumerate(data):
                yield (
                    well, tube_nr, x, y, top_depth, bottom_depth, top_height,
                      bottom_height, start, end, well_data)








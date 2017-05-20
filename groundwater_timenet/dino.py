from math import ceil

from suds.client import Client as SoapClient
from osgeo import ogr
from osgeo import osr
from owslib.wfs import WebFeatureService
from tempfile import NamedTemporaryFile

from groundwater_timenet import utils


logger = utils.setup_logging(__name__, utils.HARVEST_LOG)


WFS_URL = 'http://www.broinspireservices.nl/wfs/osgegmw-a-v1.0'
WFS_LAYER_NAME = 'gdn:Grondwateronderzoek'

SOAP_CLIENT = SoapClient("http://www.dinoservices.nl/gwservices/gws-v11?wsdl")


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
    logger.debug("Bounding Box: %d %d %d %d", minx, miny, maxx, maxy)
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
                    try_get_field(feature, 'dino_nr'),
                    try_get_field(feature, 'x_rd_crd'),
                    try_get_field(feature, 'y_rd_crd'),
                    try_get_field(feature, 'Grondwaterstand|start_date'),
                    try_get_field(feature, 'Grondwaterstand|end_date'),
                    try_get_field(feature, 'top_depth_mv'),
                    try_get_field(feature, 'bottom_depth_mv'),
                    try_get_field(feature, 'top_height_nap'),
                    try_get_field(feature, 'bottom_depth_mv'),
                )


def try_get_field(feature, fieldname, default=None):
    try:
        return feature.GetField(fieldname)
    except ValueError:
        return default


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
        raise ValueError('%s is not a valid json', source_json)
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


def load_dino_groundwater(url=WFS_URL, layer_name=WFS_LAYER_NAME):
    wfs = WebFeatureService(url=url, version='2.0.0')
    sliding_window = sliding_geom_window('NederlandRegion.json')
    for minx, miny, maxx, maxy in sliding_window:
        features = get_features(wfs, layer_name, minx, miny, maxx, maxy)
        for (well, x, y, top_depth, bottom_depth, top_height, bottom_height,
                start, end) in features:
            logger.debug(
                "Got Feature: %s %s %s %s %s %s %s %s %s", well, x, y,
                top_depth, bottom_depth, top_height, bottom_height, start, end
            )
            try:
                data = load_station_data(well)
                for i, (well_nr, tube_nr, well_data) in enumerate(data):
                    yield (
                        well, tube_nr, x, y, top_depth, bottom_depth,
                        top_height, bottom_height, start, end, well_data
                    )
            except AttributeError:
                logger.exception("Well %s doesn't contain values", well)



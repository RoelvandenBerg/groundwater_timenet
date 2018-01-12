"""
This gives a feel for the shape of the rasters the dino data and how they
intersect with the sliding window.
"""

import shutil

from osgeo import ogr, osr, gdal
import h5py
import numpy as np
import pandas as pd

import groundwater_timenet.utils
from groundwater_timenet.utils import mkdirs
from groundwater_timenet.utils import cache_h5
from groundwater_timenet.geo_utils import transform, point, bbox2polygon, sliding_geom_window
from groundwater_timenet.parse import dino
from groundwater_timenet.parse import knmi


def make_shape(filepath, fields, features, geometry_type, layername="data",
               bboxgeom=None):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(filepath)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(28992)
    if bboxgeom:
        bboxlayer = data_source.CreateLayer("BoundingBox", srs, ogr.wkbPolygon)
        bboxfeature = ogr.Feature(bboxlayer.GetLayerDefn())
        bboxfeature.SetGeometry(bboxgeom)
        bboxlayer.CreateFeature(bboxfeature)
        feature = None
    layer = data_source.CreateLayer(layername, srs, geometry_type)
    for name, field_defn in fields:
        field_name = ogr.FieldDefn(name, field_defn)
        layer.CreateField(field_name)
    for geom, values in features:
        feature = ogr.Feature(layer.GetLayerDefn())
        for i, (name, _) in enumerate(fields):
            feature.SetField(name, values[i])
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)
        feature = None
    data_source = None


def sliding_window():
    filepath = "var/data/shapes/slidinggeom/slide.shp"
    features = (
        (bbox2polygon(*coords), coords) for coords in sliding_geom_window())
    geometry_type = ogr.wkbMultiPolygon
    layername = "windows"
    fields = [(name, ogr.OFTReal) for name in ("minx", "miny", "maxx", "maxy")]
    make_shape(filepath, fields, features, geometry_type, layername)


def points_array(example_file):
    h5file = h5py.File(example_file, 'r', libver='latest')
    lat = h5file.get('lat')
    lon = h5file.get('lon')
    return np.array([
        [
            p.GetPoint() for p in (
                point(float(lon[x, y]), float(lat[x, y])) for x in
                range(350)
            ) if not transform(p)
        ] for y in range(300)
    ])


def points_list(ex_et_file, source_netcdf="var/data/cache/et_points.h5"):
    array = cache_h5(
        points_array, source_netcdf,
        example_file=ex_et_file,
    )
    points = (
        ((float(x), float(y)) for x, y, _ in array_row) for array_row in array)
    return [
        [point(*pt) for pt in points_row] for points_row in points
    ]


def knmi_et_point_cloud():
    filepath = "var/data/shapes/knmi/et_pointcloud.shp"
    et_files = groundwater_timenet.utils.raster_filenames(root="et")
    et_points = points_list(et_files[6])
    features = ((j, None) for i in et_points for j in i)
    geometry_type = ogr.wkbPoint
    fields = ()
    layername = "et"
    make_shape(filepath, fields, features, geometry_type, layername)


def knmi_rain_point_cloud():
    filepath = "var/data/shapes/knmi/rain_pointcloud.shp"
    rain_proj = osr.SpatialReference(osr.GetUserInputAsWKT(
        '+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378.14 +b=6356.75 '
        '+x_0=0 y_0=0'))
    rd_proj = osr.SpatialReference(osr.GetUserInputAsWKT('epsg:28992'))
    coord_transform = osr.CoordinateTransformation(rain_proj, rd_proj)
    affine = (0.0, 1.0, 0, -3649.9802, 0, -1.0)
    features = [
        (point(*coord_transform.TransformPoint(
            *gdal.ApplyGeoTransform(affine, column, row))[:2]), (row, column))
        for column in range(700) for row in range(765)
    ]
    geometry_type = ogr.wkbPoint
    fields = (("row", ogr.OFTReal), ("column", ogr.OFTReal))
    layername = "rain"
    bbox = transform(bbox2polygon(0.0, 48.9, 10.86, 55.97))
    make_shape(filepath, fields, features, geometry_type, layername, bbox)


def dino_point_cloud():
    filepath = "var/data/shapes/dino/dino.shp"
    dino_data = dino.list_metadata()
    OGR_TYPES = {
        'O': ogr.OFTString,
        'M': ogr.OFTString,
        'i': ogr.OFTInteger,
        'f': ogr.OFTReal
    }
    SHORT_HEADERS = {
        'bottom_depth_mv_down': 'bt_d_mv_d',
        'bottom_depth_mv_up': 'bt_d_mv_u',
        'bottom_height_nap_down': 'bt_h_nap_d',
        'bottom_height_nap_up': 'bt_h_nap_u',
        'median_step': 'median_stp',
        'top_depth_mv_down': 'top_d_mv_d',
        'top_depth_mv_up': 'top_d_mv_u',
        'top_height_nap_down': 'top_h_mv_d',
        'top_height_nap_up': 'top_h_mv_u'
    }
    fields = [
        (
            SHORT_HEADERS.get(column, column),
            OGR_TYPES[dino_data[column].dtype.kind]
        ) for column in dino_data.columns
    ]
    features = [
        [
            point(int(metadata.x), int(metadata.y)),
            [str(x) if isinstance(x, pd.Timestamp) else x for x in metadata]
        ] for _, metadata in dino_data.iterrows()
    ]
    geometry_type = ogr.wkbPoint
    layername = "dino"
    make_shape(filepath, fields, features, geometry_type, layername)


def geotop_point_cloud():
    pass


def os_clean_mkdir(delete_all=True):
    files = (
        "var/data/shapes/slidinggeom/slidinggeom.shp",
        "var/data/shapes/knmi/et_pointcloud.shp",
        "var/data/shapes/knmi/rain_pointcloud.shp",
        "var/data/shapes/geotop/geotop_pointcloud.shp",
        "var/data/shapes/dino/dino.shp",
    )
    if delete_all:
        shutil.rmtree('var/data/shapes/')
    for filepath in files:
        mkdirs(filepath)


def create_shapes():
    os_clean_mkdir()
    dino_point_cloud()
    sliding_window()
    knmi_et_point_cloud()
    knmi_rain_point_cloud()
    geotop_point_cloud()

"""
Library with common geo functions.
"""

import os
from math import ceil

from osgeo import ogr
from osgeo import osr
import numpy as np

from groundwater_timenet.utils import cache_h5


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
        source_netcdf=os.path.join("var", "data", "cache", "sliding_geom.h5")):
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
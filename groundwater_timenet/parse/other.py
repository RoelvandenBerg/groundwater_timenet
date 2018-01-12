import os

import numpy as np

import groundwater_timenet.geo_utils
from groundwater_timenet import utils
from groundwater_timenet.parse.base import SpatialVectorData, SpatialRasterData


logger = utils.setup_logging(__name__, utils.PARSE_LOG)
SOURCE_ROOT = os.path.join("var", "data", "other")


class Bofek(SpatialVectorData):
    root = "bofek"
    type = SpatialVectorData.DataType.METADATA
    spatial_driver = "OpenFileGDB"
    spatial_source_filepath = os.path.join(
        SOURCE_ROOT, 'BOFEK2012_bestandenVersie2', 'BOFEKdatabase.gdb')

    def __init__(self, *args, **kwargs):
        super(Bofek, self).__init__(*args, **kwargs)
        self._initialize_spatial_classes("BOFEK2012")
        self.empty = np.zeros(len(self.classes['bofek']))

    def _data(self, x, y, z=0, *args, **kwargs):
        point = groundwater_timenet.geo_utils.point(x, y)
        try:
            return self._layer_data("BOFEK2012", point)[0]
        except IndexError:
            return self.empty

    def _normalize(self, data):
        try:
            return self.classify('bofek', data)
        except ValueError:
            return data


class Irrigation(SpatialVectorData):
    root = "irrigation"
    type = SpatialVectorData.DataType.METADATA
    spatial_driver = "ESRI Shapefile"
    spatial_source_filepath = os.path.join(
        SOURCE_ROOT,
        'DANK005b_irrigatiewater_beregeningslocaties',
        'DANK005b_beregeningslocaties.shp'
    )
    bbox_buffer = 1000

    def _data(self, x, y, z=0, *args, **kwargs):
        bbox = groundwater_timenet.geo_utils.bbox2polygon(
            x - self.bbox_buffer,
            y - self.bbox_buffer,
            x + self.bbox_buffer,
            y + self.bbox_buffer
        )
        return np.array(
            [len([x for x in self._layer_data("GRID_CODE", bbox) if x == 1])])

    def _normalize(self, data):
        return data / 64.0


class DrinkingWater(SpatialRasterData):
    root = "drinkingwater"
    type = SpatialRasterData.DataType.METADATA
    spatial_source_filepath = os.path.join(
        SOURCE_ROOT,
        'DANK006_drinkwater',
        'DANK006_drinkwater.tif'
    )
    classes = {"drinkingwater": [0, 100, 200, 400, 600, 800, 65535]}

    def _normalize(self, data):
        return self.classify("drinkingwater", data)

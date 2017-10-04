from abc import ABCMeta, abstractproperty, abstractmethod

from osgeo import ogr, gdal
import numpy as np


class Data(object, metaclass=ABCMeta):
    class DataType:
        BASE = 0
        TEMPORAL_DATA = 1
        METADATA = 2

    nan = None
    classes = {}

    def __init__(self, convert_nan=np.nan_to_num):
        self._convert_nan = convert_nan

    def classify(self, class_type, class_name):
        classes = self.classes[class_type]
        zeros = np.zeros(len(classes))
        zeros[classes.index(class_name)] = 1
        return zeros

    @abstractproperty
    def root(self):
        return ""

    @abstractproperty
    def type(self):
        return Data.DataType.BASE

    @abstractmethod
    def _data(self, x, y, z=0, start=None, end=None):
        return

    @abstractmethod
    def _normalize(self, data):
        return np.array([])

    def _transform(self, x, y):
        return x, y

    def data(self, x, y, z=0):
        x_offset, y_offset = self._transform(x, y)
        return self._convert_nan(
            self._normalize(
                self._data(x_offset, y_offset, z)
            )
        )

    def convert_nans(self, array):
        data = array.astype("float64")
        if self.nan is not None:
            data[data == self.nan] = np.nan
        return data


class SpatialVectorData(Data, metaclass=ABCMeta):

    @abstractproperty
    def spatial_source_filepath(self):
        return ""

    @abstractproperty
    def spatial_driver(self):
        return ""

    def _use_layer(self, method, index):
        driver = ogr.GetDriverByName(self.spatial_driver)
        source = driver.Open(self.spatial_source_filepath, 0)
        return method(source.GetLayerByIndex(index))

    def _layer_data(self, field_name, geom, index=0):
        def layer_data(layer):
            layer.SetSpatialFilter(geom)
            return [feature.GetField(field_name) for feature in layer]
        return self._use_layer(layer_data, index)

    def _initialize_spatial_classes(
            self, field_name, class_type=None, index=0):
        class_type = class_type or self.root

        def spatial_classify(layer):
            self.classes[class_type] = sorted(
                list({f.GetField(field_name) for f in layer}))
        return self._use_layer(spatial_classify, index)


class SpatialRasterData(Data, metaclass=ABCMeta):

    @abstractproperty
    def spatial_source_filepath(self):
        return ""

    def __init__(self, *args, **kwargs):
        super(SpatialRasterData, self).__init__(*args, **kwargs)
        source = gdal.Open(self.spatial_source_filepath)
        self.transform = gdal.InvGeoTransform(source.GetGeoTransform())[1]
        self.array = source.ReadAsArray()

    def _transform(self, x, y):
        y, x = gdal.ApplyGeoTransform(self.transform, x, y)
        return round(x), round(y)

    def _data(self, x, y, z=0, *args, **kwargs):
        return self.array[x, y]


class TemporalData(Data, metaclass=ABCMeta):
    timedelta = "D"
    resample_method = 'first'
    type = Data.DataType.TEMPORAL_DATA

    def __init__(self, timedelta=None, resample_method=None, *args, **kwargs):
        super(TemporalData, self).__init__(*args, **kwargs)
        if timedelta is not None:
            self.timedelta = timedelta
        if resample_method is not None:
            self.resample_method = resample_method

    @abstractmethod
    def _data(self, x, y, start=None, end=None):
        return np.array([])

    def data(self, x, y, start=None, end=None):
        x_offset, y_offset = self._transform(x, y)
        data = self._data(x_offset, y_offset, start, end)
        if self.resample_method == 'first':
            data = data.resample(self.timedelta).first()
        else:
            data = data.resample(
                self.timedelta).agg(getattr(np, self.resample_method))
        data = data[start:end].as_matrix()
        return self._convert_nan(self._normalize(data))

from abc import ABCMeta, abstractproperty, abstractmethod
import datetime
import random
import operator

from osgeo import ogr, gdal
import numpy as np
import pandas as pd


class Data(object, metaclass=ABCMeta):
    class DataType:
        BASE = 0
        TEMPORAL_DATA = 1
        METADATA = 2

    nan = None
    classes = {}

    def __init__(self, convert_nan=np.nan_to_num, *args, **kwargs):
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
                self._convert_to_nans(
                    self._data(x_offset, y_offset, z)
                )
            )
        )

    def _convert_to_nans(self, array):
        if self.nan is not None:
            if isinstance(array, np.ndarray):
                data = array.astype("float64")
                data[data == self.nan] = np.nan
                return data
            else:
                return [x if x != self.nan else np.nan for x in array]
        return array


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
    first_datestamp = datetime.date(1, 1, 1)

    def __init__(
            self, timedelta=None, resample_method=None, first_datestamp=None,
            *args, **kwargs):
        super(TemporalData, self).__init__(*args, **kwargs)
        if timedelta is not None:
            self.timedelta = timedelta
        if resample_method is not None:
            self.resample_method = resample_method
        if first_datestamp is not None:
            self.first_datestamp = first_datestamp

    @abstractmethod
    def _data(self, x, y, start=None, end=None):
        return np.array([])

    def data(self, x, y, start=None, end=None):
        x_offset, y_offset = self._transform(x, y)
        return self._convert_nan(
            self._normalize(
                self._convert_to_nans(
                    self._resample(
                        self._data(x_offset, y_offset, start, end),
                        start=start, end=end
                    )
                )
            )
        )

    def _resample(self, data, start=None, end=None):
        if self.resample_method == 'first':
            data = data.resample(self.timedelta).first()
        else:
            data = data.resample(
                self.timedelta).agg(getattr(np, self.resample_method))
        if start is None and end is None:
            return data.as_matrix().T
        else:
            start = (
                start if start > self.first_datestamp else self.first_datestamp
            )
            return data.reindex(
                pd.DatetimeIndex(
                    start=start,
                    end=end,
                    freq=self.timedelta
                )
            ).as_matrix().T


class SelectorMixin(object):
    OPERATORS = {
        k: ("f", v) for k, v in (
        ("/", operator.truediv),
        ("*", operator.mul),
        ("+", operator.add),
        ("&", np.logical_and),
        ("|", np.logical_or),
        ("=", operator.eq),
        (">", operator.gt),
        ("<", operator.lt),
        ("<=", operator.le),
        (">=", operator.ge)
    )}

    def select(self, selection, dataframe):
        if not selection:
            return dataframe
        columns = {c: ("v", dataframe[c]) for c in dataframe.columns.tolist()}
        columns.update(self.OPERATORS)
        selected = [
            self._get_operator(x, columns) for x in
            selection.replace("(", " ( ").replace(")", " ) ").split(' ') if x
        ]
        return dataframe[self._parse_selected_part(selected)]

    def _parse_selected_part(self, selected_part):
        if len(selected_part) == 1:
            return selected_part[0][1]
        leftmost_type, leftmost_argument = selected_part.pop(0)
        if leftmost_type == 'f':
            raise ValueError("Function without previous argument.")
        if leftmost_type == 'b':
            if leftmost_argument == ")":
                raise ValueError("Brackets in wrong order.")
            selected_part = self._parse_selected_part(
                self._parse_brackets(selected_part)
            )
        if leftmost_type == 'v':
            function_type, function_ = selected_part.pop(0)
            assert function_type == 'f'
            selected_part = function_(
                leftmost_argument, self._parse_selected_part(selected_part))
        return selected_part

    def _parse_brackets(self, selected_part):
        bracketscore = 1
        between_brackets = []
        while bracketscore > 0:
            try:
                leftmost_type, leftmost_argument = selected_part.pop(0)
            except IndexError:
                raise ValueError("Missing ending bracket.")
            if leftmost_type == 'b':
                if leftmost_argument == '(':
                    bracketscore += 1
                if leftmost_argument == ')':
                    bracketscore -= 1
            else:
                between_brackets.append((leftmost_type, leftmost_argument))
        return [
                   ('v', self._parse_selected_part(between_brackets))
               ] + selected_part

    @staticmethod
    def _get_operator(x, columns):
        res = columns.get(x)
        if res:
            return res
        try:
            return 'v', float(x)
        except:
            return 'b', x


class BaseData(SelectorMixin, TemporalData, metaclass=ABCMeta):
    data = None

    def __init__(
            self, seed=4177, train_percentage=70, validation_percentage=20,
            test_percentage=10, selection="", *args, **kwargs):
        assert(
            test_percentage + validation_percentage + train_percentage == 100)
        super(BaseData, self).__init__(*args, **kwargs)
        self.selection = selection
        self.seed = seed
        self._all_metadata = self._read_metadata()
        self._parts = {
            "train": self._pct_to_index(0, train_percentage),
            "validation": self._pct_to_index(
                train_percentage, validation_percentage),
            "test": self._pct_to_index(validation_percentage, 100)
        }
        self._iterator = None

    def _pct_to_index(self, from_pct, to_pct):
        return slice(
            int(len(self) * from_pct / 100.0), int(len(self) * to_pct / 100.0)
        )

    @abstractmethod
    def _data(self, slice_):
        yield

    @abstractmethod
    def _read_metadata(self):
        return []

    def __len__(self):
        return len(self._all_metadata)

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is not None:
            x, y, z, meta_row, metadata, dataframe = next(self._iterator)
            start = meta_row.start.to_pydatetime().date()
            end = meta_row.end.to_pydatetime().date()
            return (
                x, y, z, start, end, metadata,
                self._convert_nan(
                    self._normalize(
                        self._convert_to_nans(
                            self._resample(dataframe, start, end)
                        ))))
        raise StopIteration

    def __call__(self, part):
        random.seed(self.seed)
        self._iterator = iter(self._data(self._parts[part]))
        return iter(self)



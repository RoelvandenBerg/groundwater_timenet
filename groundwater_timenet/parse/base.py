from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np


class Data(object, metaclass=ABCMeta):
    class DataType:
        BASE = 0
        TEMPORAL_DATA = 1
        METADATA = 2

    nan = None

    def __init__(self, convert_nan=np.nan_to_num):
        self._convert_nan = convert_nan

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

    def data(self, x, y, z=0):
        return self._convert_nan(self._normalize(self._data(x, y, z)))

    def convert_nans(self, array):
        data = array.astype("float64")
        if self.nan is not None:
            data[data == self.nan] = np.nan
        return data


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
        data = self._data(x, y, start, end)
        if self.resample_method == 'first':
            data = data.resample(self.timedelta).first()
        else:
            data = data.resample(
                self.timedelta).agg(getattr(np, self.resample_method))
        data = data[start:end].as_matrix()
        return self._convert_nan(self._normalize(data))

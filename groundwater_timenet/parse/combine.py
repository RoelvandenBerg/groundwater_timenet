import h5py
import datetime

from .base import Data
from .geotop import GeotopData
from .knmi import WeatherStationData, RainData, EvapoTranspirationData
from .dino import DinoData


class DataCombiner(object):
    """
    For different timesteps see:
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    """
    timedeltas = {
        'hour': "H",
        'day': "D",
        'week': "W",
        'month': "M",
        'halfmonthly': "SM"
    }
    data_sources = (
        WeatherStationData,
        RainData,
        EvapoTranspirationData,
        GeotopData,
        DinoData
    )

    def __init__(self, timestep="day", origin=None):
        self.origin = datetime.datetime(
            1970, 1, 1) if origin is None else origin
        self.timestep = self.timedeltas.get(timestep, timestep)
        self.metadata = self._filter_source(Data.DataType.METADATA)
        self.temporaldata = self._filter_source(Data.DataType.TEMPORAL_DATA)
        self.basedata = self._filter_source(Data.DataType.BASE)[0]

    def _filter_source(self, data_type):
        return tuple(
            filter(lambda ds: ds.type == data_type, self.data_sources))

    @property
    def dino_filepaths(self):
        return ()
        # return dino.filepaths()

    def combine(self):
        for filepath in self.dino_filepaths:
            with h5py.File(filepath, 'r', libver='latest') as h5file:
                for metadata in h5file["metadata"]:
                    x, y = metadata[2:4].astype('int')
    #
    # def add_h5dataset(self, data, h5_file, target_dataset):
    #     dataset = h5_file.create_dataset(
    #         target_dataset,
    #         data.shape,
    #         dtype=data.dtype)
    #     dataset[...] = data

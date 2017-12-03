import datetime
import os

import numpy as np

from .base import Data
from .geotop import GeotopData
from .knmi import WeatherStationData, RainData, EvapoTranspirationData
from .dino import DinoData
from groundwater_timenet import utils


logger = utils.setup_logging(__name__, utils.PARSE_LOG, "INFO")


DEFAULT_SELECTION = (
    '((counts / (days / 15)) > 0) & '
    '(median_step <= 15) & '
    '(days > (365 * 2))'
)
FIRST_DATESTAMP = datetime.date(1965, 1, 1)


class Combiner(object):
    """
    For different timesteps see:
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    """
    timedeltas = {
        'hour': "H",
        'day': "D",
        'week': "W",
        'month': "M",
        'halfmonthly': "SM",
        '15day': '15D'
    }
    data_sources = (
        WeatherStationData,
        RainData,
        EvapoTranspirationData,
        GeotopData,
        DinoData
    )

    def __init__(
            self, timestep="15day", resample_method='first',
            first_datestamp=FIRST_DATESTAMP, chunk_size=1000,
            selection=DEFAULT_SELECTION, *args, **kwargs):
        self.chunk_size = chunk_size
        self.timestep = self.timedeltas.get(timestep, timestep)
        self._meta_data = [metadata(*args, **kwargs)  for metadata in
                         self._filter_source(Data.DataType.METADATA)]
        self._temporal_data = [
            temporal(
                timedelta=self.timestep, resample_method=resample_method,
                first_timestamp=first_datestamp, *args, **kwargs)
            for temporal in self._filter_source(Data.DataType.TEMPORAL_DATA)
        ]
        self._base_data = self._filter_source(Data.DataType.BASE)[0](
            timedelta=self.timestep, resample_method=resample_method,
            selection=selection, first_timestamp=first_datestamp,
            *args, **kwargs
        )
        self.dataset_name = tuple(
            name + "_" + str(i) for name in ("base", "temporal", "meta")
            for i in range(self.chunk_size)
        )

    def _filter_source(self, data_type):
        return tuple(
            filter(lambda ds: ds.type == data_type, self.data_sources))

    def meta_data(self, base, x, y, z):
        return np.concatenate(
            [base] + [metadata.data(x, y, z) for metadata in self._meta_data])

    def temporal_data(self, base, x, y, start, end):
        temporal_data = [base] + [
                data.data(x, y, start, end)
                for data in self._temporal_data
            ]
        return np.vstack(temporal_data)

    def combine(self, part):
        temporal = []
        meta = []
        base = []
        for i, params in enumerate(self._base_data(part)):
            x, y, z, start, end, base_metadata, base_data = params
            base.append(base_data)
            temporal.append(self.temporal_data(base_data, x, y, start, end))
            meta.append(self.meta_data(base_metadata, x, y, z))
            if not i % self.chunk_size and i != 0:
                filepath = os.path.join(
                    "var", "data", "neuralnet", part, str(i) + ".h5")
                utils.store_h5(
                    data=base + temporal + meta,
                    dataset_name=self.dataset_name,
                    target_h5=filepath,
                    many=True
                )
                logger.info(
                    "Combined %d series in total. Wrote %d to file %s.",
                    i, self.chunk_size, filepath)
                temporal = []
                meta = []
                base = []

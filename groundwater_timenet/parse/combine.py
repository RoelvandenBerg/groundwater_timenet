from itertools import chain
import datetime
import time

import numpy as np

from .base import Data
from .geotop import GeotopData
from .knmi import WeatherStationData, RainData, EvapoTranspirationData
from .dino import DinoData
from .other import Bofek, Irrigation, DrinkingWater
from groundwater_timenet import utils
from groundwater_timenet.learn.generator import ConvCombinerGenerator
from groundwater_timenet.learn.settings import *


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
        'hour': ("H", datetime.timedelta(hours=1)),
        'day': ("D", datetime.timedelta(days=1)),
        'week': ("W", datetime.timedelta(days=7)),
        'month': ("M", datetime.timedelta(days=30, hours=12)),
        'halfmonthly': ("SM", datetime.timedelta(days=15)),
        '15day': ("15D", datetime.timedelta(days=15)),
    }
    data_sources = (
        DinoData,
        WeatherStationData,
        RainData,
        EvapoTranspirationData,
        GeotopData,
        Bofek,
        Irrigation,
        DrinkingWater
    )

    # TODO: change to '15day'?
    def __init__(
            self, timestep="halfmonthly", resample_method='first',
            first_datestamp=FIRST_DATESTAMP, chunk_size=1000,
            selection=DEFAULT_SELECTION, *args, **kwargs):
        self.chunk_size = chunk_size
        self.timestep = self.timedeltas.get(timestep, timestep)[0]
        self.temporal_shift = self.timedeltas.get(timestep, timestep)[1]
        self._meta_data = [metadata(*args, **kwargs) for metadata in
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
        start += self.temporal_shift
        end += 2 * self.temporal_shift
        base_length = base.shape[0] - 1
        temporal_data = [base[:base_length]] + [
            data.data(x, y, start, end)[:base_length] for data in
            self._temporal_data
        ]
        return np.hstack(temporal_data)

    def combine(self, part):
        temporal = []
        meta = []
        base = []
        for i, params in enumerate(self._base_data(part)):
            x, y, z, start, end, base_metadata, base_data = params
            base.append(base_data[1:])
            temporal.append(
                self.temporal_data(base_data, x, y, start, end))
            meta.append(self.meta_data(base_metadata, x, y, z))
            if not (i + 1) % self.chunk_size and i != 0:
                filepath = os.path.join(
                    "var", "data", "neuralnet", part, str(i + 1) + ".h5")
                utils.store_h5(
                    data=base + temporal + meta,
                    dataset_name=self.dataset_name,
                    target_h5=filepath,
                    many=True
                )
                logger.info(
                    "Combined %d series in total. Wrote %d to file %s.",
                    i + 1, self.chunk_size, filepath)
                temporal = []
                meta = []
                base = []


class UncompressedCombiner(Combiner):

    def __init__(
            self, timestep="halfmonthly", resample_method='first',
            first_datestamp=FIRST_DATESTAMP, chunk_size=CHUNK_SIZE,
            selection=DEFAULT_SELECTION, base="neuralnet", data_type="train",
            batch_size=BATCH_SIZE, meta_size=META_SIZE, temporal_size=TEMPORAL_SIZE,
            input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, *args, **kwargs):
        super().__init__(
            timestep="halfmonthly", resample_method='first',
            first_datestamp=FIRST_DATESTAMP, chunk_size=chunk_size,
            selection=DEFAULT_SELECTION, *args, **kwargs)
        self.generator = ConvCombinerGenerator(
            base, data_type, batch_size, chunk_size, meta_size, temporal_size,
            input_size, output_size)
        self.dataset_name = tuple(
            name + "_" + str(i) for name in ("input", "output")
            for i in range(self.chunk_size)
        )

    def combine(self, part):
        logger.info(
            '################# Start combining %s data. #################',
            part
        )
        size_ = self.chunk_size * self.generator.batch_size / 100
        i = 0
        total = 0
        start_time = time.time()
        for j, params in enumerate(self._base_data(part)):
            x, y, z, start, end, base_metadata, base = params
            temporal = self.temporal_data(base, x, y, start, end)
            meta = self.meta_data(base_metadata, x, y, z)
            if not self.generator.generate_batch(base[1:], meta, temporal):
                logger.warn('Empty series at position %d', i)
                continue
            duration = time.time() - start_time
            total = max(total, len(self._base_data))
            time_left_seconds = (duration / (j + 1)) * (total - j)
            time_left_hours = time_left_seconds // 3600
            time_left_minutes = (time_left_seconds % 3600) // 60
            logger.info(
                "Packed series #%d. At: %d  |  ETA: %d:%d" % (
                    j, round(self.generator.input_data.shape[0] / size_, 2),
                    time_left_hours, time_left_minutes
                )
            )
            for input_data, output_data in self.generator.unpack_batches(
                    chunk_size=self.chunk_size):
                i += 1
                filepath = os.path.join(
                    # "var", "data", "neuralnet", part, str(i) + ".h5")
                    "neuralnet", part, str(i + 1) + ".h5")
                utils.store_h5(
                    data=chain.from_iterable([input_data, output_data]),
                    dataset_name=self.dataset_name,
                    target_h5=filepath,
                    many=True
                )
                logger.info(
                    "Combined %d series in total. Wrote %d to file %s.",
                    i + 1, self.chunk_size, filepath)

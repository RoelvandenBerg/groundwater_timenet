import datetime
import os
import random

import h5py
import numpy as np
import pandas as pd

from groundwater_timenet import utils
from groundwater_timenet.collect import dino as collect
from groundwater_timenet.parse.base import BaseData


logger = utils.setup_logging(__name__, utils.PARSE_LOG, "INFO")

DATETIME_EPOCH = datetime.datetime(1970,1,1)


def filepaths():
    return (
        os.path.join(root, f) for root, dirs, files in
        os.walk('var/data/dino') for f in files if f.endswith('hdf5')
    )


def _list_metadata(shuffled=False):
    base = os.path.join(utils.DATA, collect.FILENAME_BASE)
    logger.info("All y coordinates: %s", str(os.listdir(base)))
    total = set()
    for filepath in filepaths():
        h5_file = h5py.File(filepath, "r")
        metadata = h5_file.get("metadata", [])
        new_meta = set()
        for md in metadata:
            wellcode = md[0].decode('utf8')
            filtercode = md[1].decode('utf8')
            dataset = h5_file.get(wellcode + filtercode)
            try:
                s, e = dataset[[-1, 0], 0].astype('datetime64[D]')
            except OSError:
                logger.debug("Left out well %s.%s: only 1 record found",
                             wellcode, filtercode)
                continue
            days = int((e - s) / 86400)
            if days < (365 * 2):
                logger.debug("Left out well %s.%s: only %d records found",
                             wellcode, filtercode, days)
                continue
            density = dataset.shape[0] / days
            delta = abs(np.diff(dataset[:,0]))
            min_step = int(np.min(delta) / 86400)
            max_step = int(np.max(delta) / 86400)
            median_step = int(np.median(delta) / 86400)
            new_meta.add(
                tuple([filepath.encode('utf8')] +
                      md.tolist() +
                      [days, dataset.shape[0], density, s, e, min_step,
                       median_step, max_step])
            )
        total = total.union(new_meta)
        length = len(new_meta)
        if length == 0:
            message = "File " + filepath + "doesn't contain metadata"
        else:
            message = "File " + filepath + "contains " + str(length) + \
                      " records"
        logger.info(message)
    logger.info("Total records found: %d", len(total))
    total = sorted([list(x) for x in total])
    if shuffled:
        random.shuffle(total)
    result = np.array(total)
    return result


def list_metadata(shuffled=False):
    metadata = utils.cache_h5(
        _list_metadata,
        target_h5=os.path.join("var", "data", "cache", "dino_base_metadata"),
        shuffled=shuffled
    )
    metadata[metadata == b''] = np.nan
    columns = [
        ('filepath', str),
        ('wellcode', str),
        ('filtercode', str),
        ('x', int),
        ('y', int),
        ('start_meta', str),
        ('end_meta', str),
        ('top_depth_mv_up', float),
        ('top_depth_mv_down', float),
        ('bottom_depth_mv_up', float),
        ('bottom_depth_mv_down', float),
        ('top_height_nap_up', float),
        ('top_height_nap_down', float),
        ('bottom_height_nap_up', float),
        ('bottom_height_nap_down', float),
        ('days', int),
        ('counts', int),
        ('density', float),
        ('start', 'datetime64[s]'),
        ('end', 'datetime64[s]'),
        ('min_step', int),
        ('median_step', int),
        ('max_step', int)
    ]
    values = {}
    for i, (column, astype) in enumerate(columns):
        try:
            values[column] = metadata[:, i].astype(astype)
        except ValueError:
            values[column] = metadata[:, i].astype(float).astype(astype)

    return pd.DataFrame(values)


class DinoData(BaseData):
    root = 'dino'
    type = BaseData.DataType.BASE
    relevant_meta = (
        'top_depth_mv_up',
        'top_depth_mv_down',
        'bottom_depth_mv_up',
        'bottom_depth_mv_down',
        'top_height_nap_up',
        'top_height_nap_down',
        'bottom_height_nap_up',
        'bottom_height_nap_down'
    )

    def _read_metadata(self):
        return list_metadata(shuffled=True)

    def _data(self, slice_):
        metadata_sorted = self.select(
            self.selection, self._all_metadata[slice_]
        ).copy().sort_values(by="days", ascending=False)
        self._length = len(metadata_sorted)
        filtercodes = {
            s: i for i, s in enumerate(sorted(set(metadata_sorted.filtercode)))
        }
        metadata = (a for _, a in metadata_sorted.iterrows())

        for row in metadata:
            h5_file = h5py.File(row.filepath, "r")
            timeseries_code = row.wellcode + row.filtercode
            data = h5_file.get(timeseries_code, [])
            index = pd.DatetimeIndex(data[:, 0].astype('datetime64[s]'))
            dataframe = pd.DataFrame(data[:, 1], index=index)
            z = (
                row.top_height_nap_up or
                row.top_height_nap_down or
                row.bottom_height_nap_up or
                row.bottom_height_nap_down or
                -9999
            )
            yield row.x, row.y, z, row, self.metadata_array(
                row, filtercodes), dataframe
            self._length -= 1
        self._length = None

    def _to_date(self, date):
        datestr = str(date)
        return datetime.datetime(datestr[:4], datestr[4:6], datestr[6:])

    def count_timesteps(self):
        total = sum([y[1].shape[0] for y in self._data(slice(None))])
        logger.info("%d individual timesteps found", total)
        return total

    def _filtercode(self, code, filtercodes):
        zeros = np.zeros(56)
        zeros[filtercodes[code]] = 1
        return zeros

    def metadata_array(self, row, filtercodes):
        return self._nan_to_num(
            np.concatenate(
                [self._filtercode(row.filtercode, filtercodes), np.array(
                    [row[meta] for meta in self.relevant_meta])]
            )
        )

    def _normalize(self, data):
        return data / 500  # This can also be normalized using a distribution.

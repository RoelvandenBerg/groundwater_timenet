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
        new_meta = {
            tuple([filepath.encode('utf8')] + list(x)) for x in metadata}
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
    return [
        (meta[0], [meta[1], meta[2], int(meta[3]), int(meta[4]), meta[5],
                   meta[6]] + [utils.tryfloat(x) for x in meta[7:]])
        for meta in ([d.decode('utf-8') for d in md] for md in metadata)
    ]


class DinoData(BaseData):
    root = 'dino'
    type = BaseData.DataType.BASE

    def _read_metadata(self):
        return list_metadata(shuffled=True)

    def _data(self, slice_):
        for filepath, metadata in self._all_metadata[slice_]:
            h5_file = h5py.File(filepath, "r")
            data = h5_file.get(metadata[0] + metadata[1], [])
            yield data, metadata

    def _unpack(self, data, metadata):
        x, y = metadata[2:4]
        z = metadata[10] or metadata[11] or 1  # = top_height_nap
        index = pd.DatetimeIndex(data[:,0].astype('datetime64[s]'))
        dataframe = pd.DataFrame(data[:,1], index=index)
        return x, y, z, metadata[6:], dataframe

    def _to_date(self, date):
        datestr = str(date)
        return datetime.datetime(datestr[:4], datestr[4:6], datestr[6:])

    def count_timesteps(self):
        total = sum([y[1].shape[0] for y in self._data(slice(None))])
        logger.info("%d individual timesteps found", total)
        return total

    def _normalize(self, data):
        return data / 500  # This can also be normalized using a distribution.

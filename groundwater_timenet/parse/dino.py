import os
import random

import h5py
import numpy as np

from groundwater_timenet import utils
from groundwater_timenet.download import dino as download
from groundwater_timenet.parse.base import Data


logger = utils.setup_logging(__name__, utils.PARSE_LOG, "INFO")


def list_metadata(shuffled=False):
    base = os.path.join(utils.DATA, download.FILENAME_BASE)
    logger.info("All y coordinates: %s", str(os.listdir(base)))
    total = set()
    for root, dirs, files in os.walk(base):
        hdf5_files = [
            os.path.join(root, hdf5_file) for hdf5_file in files
            if hdf5_file[-4:] == "hdf5"
        ]
        for filepath in hdf5_files:
            h5_file = h5py.File(filepath, "r")
            metadata = h5_file.get("metadata", [])
            new_meta = {
                (filepath, tuple(d.decode('utf-8') for d in x))
                for x in metadata
            }
            total = total.union(new_meta)
            length = len(new_meta)
            if length == 0:
                message = "File " + filepath + "doesn't contain metadata"
            else:
                message = "File " + filepath + "contains " + str(length) + \
                          " records"
            logger.info(message)
    logger.info("Total records found: %d", len(total))
    total = list(total)
    if shuffled:
        random.shuffle(total)
    return total


def try_float(x, metadata_raw):
    try:
        return float(x)
    except ValueError:
        if x != '':
            logger.exception("could not covert {} to float".format(x))
        return np.nan


def random_stations(from_pct=0, to_pct=100):
    all_data = list_metadata(shuffled=True)
    length = len(all_data)
    from_ = int(length * from_pct / 100.0)
    to_ = int(length * to_pct / 100.0)
    for filepath, metadata_binary in all_data[from_:to_]:
        h5_file = h5py.File(filepath, "r")
        metadata_raw = [x.decode('utf8') for x in metadata_binary]
        metadata = (
            metadata_raw[:2] + [int(x) for x in metadata_raw[2:4]] +
            metadata_raw[4:6] + [try_float(x, metadata_raw)
                                 for x in metadata_raw[6:]]
        )
        data = h5_file.get(metadata[0] + metadata[1], [])
        yield metadata, data


def count():
    total = sum([y[1].shape[0] for y in random_stations()])
    logger.info("%d individual timesteps found", total)
    return total


def filepaths():
    return (
        os.path.join(root, f) for root, dirs, files in
        os.walk('var/data/dino') for f in files if f.endswith('hdf5')
    )


class DinoData(Data):
    type = Data.DataType.BASE

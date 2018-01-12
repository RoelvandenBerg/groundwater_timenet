"""
Library with common functions.
"""

import logging
import os

import h5py
import numpy as np

PARSE_LOG = os.path.join('var', 'log', 'parse.log')
HARVEST_LOG = os.path.join('var', 'log', 'harvest.log')
DATA = os.path.join('var', 'data')


def mkdirs(path):
    """Create a directory for a path if it doesn't exist yet."""
    dirname = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass


def setup_logging(name, filename, level="DEBUG"):
    mkdirs(filename)
    logging.basicConfig(
        filename=filename,
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)
    return logger


logger = setup_logging(__name__, PARSE_LOG, level="DEBUG")


def store_h5(
        data, dataset_name, target_h5=os.path.join("var", "data", "cache", "cache.h5"), many=False):
    if not many:
        data = [data]
        dataset_name = [dataset_name]
    mkdirs(target_h5)
    with h5py.File(target_h5, "w", libver='latest') as h5_file:
        for i, dataset_data in enumerate(data):
            dataset = h5_file.create_dataset(
                dataset_name[i],
                dataset_data.shape,
                dtype=dataset_data.dtype)
            dataset[...] = dataset_data


def read_h5(filepath, dataset_name, index=None, many=False):
    with h5py.File(filepath, 'r', libver='latest') as h5file:
        if not many:
            if index is None:
                index = ()
            return h5file.get(dataset_name)[index]
        else:
            if index is None:
                index = [() for _ in range(len(dataset_name))]
            return tuple(
                h5file.get(name)[index[i]] for i, name in
                enumerate(dataset_name)
            )


def cache_h5(source_data_function, target_h5, cache_dataset_name=None,
             decode=None, *source_data_function_args,
             **source_data_function_kwargs
             ):
    cache_dataset_name = cache_dataset_name or os.path.basename(
        target_h5).strip('.h5')
    if not os.path.exists(target_h5):
        mkdirs(target_h5)
        data = source_data_function(
            *source_data_function_args,
            **source_data_function_kwargs
        )
        store_h5(data, cache_dataset_name, target_h5)
    with h5py.File(target_h5, "r", libver='latest') as target:
        return target[cache_dataset_name][()]


def parse_filepath(minx, miny, filename_base="dino"):
    filepath = os.path.join(
        DATA, filename_base, str(miny), str(minx) + ".hdf5"
    )
    mkdirs(filepath)
    return filepath


def int_or_nan(x):
    try:
        return int(x)
    except ValueError:
        return np.nan


def try_h5(fn, d=None):
    try:
        f = h5py.File(fn, 'r')
        if d is not None:
            f.get(d)[:]
    except (OSError, TypeError):
        return fn


def _get_raster_filenames(rootdir, raise_errors, dataset_name=None):
    files = sorted(
        [
            os.path.join(r, f[0]) for r, d, f in os.walk('var/data/' + rootdir)
            if f
        ]
    )
    faulty = [x for x in [try_h5(f, dataset_name) for f in files] if x]
    if faulty:
        message = 'Broken HDF5 files found:\n- {}'.format('\n- '.join(faulty))
        if raise_errors:
            raise OSError(message)
        else:
            logger.debug(message)
            return np.array(
                [f.encode('utf8') for f in files if f not in faulty])
    return np.array([f.encode('utf8') for f in files])


def raster_filenames(
        root, source_netcdf=None, raise_errors=True, dataset_name=None):
    source_netcdf = source_netcdf or os.path.join("var", "data", "cache", "{}files.h5".format(
        root if raise_errors else "filtered_" + root))
    filenames = cache_h5(
        _get_raster_filenames, source_netcdf,
        cache_dataset_name=root,
        rootdir=root,
        raise_errors=raise_errors,
        dataset_name=dataset_name
    )
    return [f.decode('utf8') for f in filenames]

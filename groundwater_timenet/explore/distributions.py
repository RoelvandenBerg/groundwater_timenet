"""
The distributions of the datasets and the functions that deliver them.

These functions are here merely for completeness and as a basis for further
tweaking when more detailed information is required. The generated
distributions can be found in exploration/distributions and the charts in
exploration/images.
"""

import json
import os

from matplotlib import pyplot
from netCDF4 import Dataset
import h5py
import numpy as np

from groundwater_timenet import utils
from groundwater_timenet import parse
from groundwater_timenet.utils import raster_filenames
from abc import abstractproperty, abstractmethod, ABCMeta


class Counts(dict, metaclass=ABCMeta):

    def __init__(self, cache_json=False, use_cache=True, count=False):
        self._cache = cache_json
        if use_cache and os.path.exists(self.json_filepath):
            with open(self.json_filepath, 'r') as json_file:
                self.update(json.loads(json_file.read()))
        if count:
            self.count()

    @abstractmethod
    def dataset_generator(self, key):
        pass

    @abstractproperty
    def json_filename(self):
        pass

    @abstractproperty
    def dataset_names(self):
        pass

    @property
    def json_filepath(self):
        return os.path.join("exploration", "distributions", self.json_filename)

    def _unique_counts(self, key):
        # we iterate over the file in 100-size steps over the j-axis since the
        # file is to large to handle at once in memory
        all_unique = [
            (str(value), counts[ii]) for values, counts in [
                np.unique(dataset, return_counts=True) for dataset in
                self.dataset_generator(key)
            ] for ii, value in enumerate(values)
        ]
        unique_counts = {}
        for k, v in all_unique:
            unique_counts[k] = unique_counts.get(k, 0) + v
        return unique_counts

    def count(self):
        """
        We count all values for the relevant datasets. And store them in a
        json.

        Returns:
            dictionary of dataset counts.
        """
        self.update({
            name: self._unique_counts(name) for name in self.dataset_names
        })
        if self._cache:
            self.cache()

    def cache(self):
        """
        Caches dictionary to json.
        """
        # somehow the generated dictionary is not correct
        strfied = "{\n    " + ",\n    ".join(
            ['"' + str(k) + '": ' + str(v).replace("'", '"') for k, v in
             self.items()]) + "\n}"
        with open(self.json_filepath, 'w') as f:
            f.write(strfied)


class Plots(Counts, metaclass=ABCMeta):
    @abstractproperty
    def png_base(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    def _count_plot(
            self, x_axis, data, legend=None, xlabel=None, method='plot'):
        pyplot.cla()
        for data_set in data:
            getattr(pyplot, method)(x_axis, data_set)
        if xlabel is not None:
            pyplot.xlabel(xlabel)
            filename = xlabel
        if legend is not None:
            pyplot.legend(legend)
            filename = legend[0]
        filepath = os.path.join(
            'exploration', 'images', self.png_base + '_' + filename + '.png')
        pyplot.ylabel('count')
        fig = pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(filepath, dpi=100)

    def _create_data(self, classes, headers):
        sorted_classes = sorted(classes)
        return [
            [self[h].get(str(c), 0) for c in sorted_classes] for h in headers
        ]

    def percent_plot(self, headers, start=1, end=100, xlabel='percent'):
        x_axis = list(range(start, end))
        data = self._create_data(x_axis, headers)
        self._count_plot(x_axis, data, headers, xlabel)

    def class_plot(self, label):
        classes = sorted([int(c) for c in self[label].keys() if c != "--"])
        self._count_plot(
            x_axis=classes,
            data=self._create_data(classes, (label,)),
            xlabel=label,
            method="bar")

    def line_plot(self, keys):
        end = max([int(k) for key in keys for k in self[key].keys()])
        start = min([int(k) for key in keys for k in self[key].keys()])
        self.percent_plot(keys, start=start, end=end, xlabel=keys[0])


class Geotop(Plots):
    """
    Object with geotop counts.

    Running this for the first time will take time.
    When store_json is set to True and the file does not exist, it will save a
    json with the counts to disk (exploration/distributions/geotop.json).

    Has a plot method that shows a series of mapnik plots of the distributions.
    """
    json_filename = "geotop.json"
    dataset_names = parse.geotop.RELEVANT_VARIABLES
    png_base = 'geotop'

    def __init__(self, *args, **kwargs):
        super(Geotop, self).__init__(*args, **kwargs)
        filepath = os.path.join(utils.DATA, 'geotop', "geotop.nc")
        self.geotop = Dataset(filepath, "r")

    def dataset_generator(self, key):
        for j in range(0, self.geotop[key].shape[1], 100):
            yield self.geotop[key][:, j:j + 100, :]

    def plot(self):
        self.percent_plot(tuple('kans_' + str(j + 1) for j in range(9)))
        self.percent_plot(('onz_lk', 'onz_ls'))
        self.class_plot('strat')
        self.class_plot('lithok')


class Knmi(Plots):
    """
    Object with KNMI rain and evapotranspiration counts.

    Running this for the first time will take time.
    When store_json is set to True and the file does not exist, it will save a
    json with the counts to disk (exploration/distributions/geotop.json).

    Has a plot method that shows a series of mapnik plots of the distributions.
    """
    json_filename = 'knmi.json'
    dataset_names = ('et', 'rain')
    subdataset_name = {"et": 'prediction', 'rain': 'image1/image_data'}
    fraction = {'et': 10, 'rain': 0.01}
    png_base = 'knmi'

    def __init__(self, *args, **kwargs):
        super(Knmi, self).__init(*args, **kwargs)
        self.filenames = {
            'et': raster_filenames(root='et'),
            'rain': raster_filenames(root='rain'),
        }

    def dataset_generator(self, key):
        for filepath in self.filenames[key]:
            yield (utils.read_h5(filepath, self.subdataset_name[key]) *
                   self.fraction[key]).astype('int')

    def plot(self):
        for key in self.dataset_names:
            if self[key]:
                self.percent_plot((key, ), xlabel=key)


class Dino(Plots):
    """
    Object with Dino counts.

    Running this for the first time will take time.
    When store_json is set to True and the file does not exist, it will save a
    json with the counts to disk (exploration/distributions/geotop.json).

    Has a plot method that shows a series of mapnik plots of the distributions.
    """
    json_filename = 'dino.json'
    dataset_names = (
        'groundwater',
        'top_depth_mv_up', 'top_depth_mv_down',
        'bottom_depth_mv_up', 'bottom_depth_mv_down',
        'top_height_nap_up', 'top_height_nap_down',
        'bottom_height_mv_up', 'bottom_height_mv_down'
    )
    dataset_indexes = {
        'top_depth_mv_up': 6,
        'top_depth_mv_down': 7,
        'bottom_depth_mv_up': 8,
        'bottom_depth_mv_down': 9,
        'top_height_nap_up': 10,
        'top_height_nap_down': 11,
        'bottom_height_mv_up': 12,
        'bottom_height_mv_down': 13
    }
    png_base = 'dino'

    @property
    def filepaths(self):
        return parse.dino.filepaths()

    def groundwater_generator(self):
        for filepath in self.filepaths:
            with h5py.File(filepath, 'r', libver='latest') as h5file:
                yield np.concatenate(
                    [h5file[k][:, 1] for k in list(h5file) if k != "metadata"]
                ).astype('int')

    def metadata_generator(self, key):
        k = self.dataset_indexes[key]
        for filepath in self.filepaths:
            with h5py.File(filepath, 'r', libver='latest') as h5file:
                metadata = h5file['metadata'][:, k]
                yield metadata[metadata != b''].astype('float64').astype('int')

    def dataset_generator(self, key):
        if key == 'groundwater':
            return self.groundwater_generator()
        return self.metadata_generator(key)

    def plot(self):
        if self['groundwater']:
            self.line_plot(('groundwater',))
        if self['top_depth_mv_up']:
            self.line_plot(('top_depth_mv_up', 'top_depth_mv_down',))
        if self['bottom_depth_mv_up']:
            self.line_plot(('bottom_depth_mv_up', 'bottom_depth_mv_down',))
        if self['top_height_nap_up']:
            self.line_plot(('top_height_nap_up', 'top_height_nap_down',))
        if self['bottom_height_mv_up']:
            self.line_plot(('bottom_height_mv_up', 'bottom_height_mv_down'))

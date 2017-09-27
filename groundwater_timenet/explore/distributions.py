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
import numpy as np

from groundwater_timenet import utils
from groundwater_timenet.parse.geotop import RELEVANT_VARIABLES
from groundwater_timenet.download.knmi import raster_filenames
from abc import abstractproperty, abstractmethod, ABCMeta


class Counts(dict, metaclass=ABCMeta):

    def __init__(self, cache_json=False):
        self._cache = cache_json
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

    def count(self, use_cache=True):
        """
        We count all values for the relevant datasets. And store them in a
        json.

        Returns:
            dictionary of dataset counts.
        """
        if use_cache and os.path.exists(self.json_filepath):
            with open(self.json_filepath, 'r') as json_file:
                self.update(json.loads(json_file.read()))
        else:
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
            ['"' + k + '": ' + str(v).replace("'", '"') for k, v in
             self.items()]) + "\n}"
        with open(self.json_filepath, 'w') as f:
            f.write(strfied)


class Plots(Counts, metaclass=ABCMeta):

    @abstractmethod
    def plot(self):
        pass

    @staticmethod
    def _count_plot(x_axis, data, legend=None, xlabel=None, method='plot'):
        for data_set in data:
            getattr(pyplot, method)(x_axis, data_set)
        if legend is not None:
            pyplot.legend(legend)
        if xlabel is not None:
            pyplot.xlabel(xlabel)
        pyplot.ylabel('count')
        pyplot.show()

    def _create_data(self, classes, headers):
        sorted_classes = sorted(classes)
        return [
            [self[h].get(str(c), 0) for c in sorted_classes] for h in headers
        ]

    def percent_plot(self, headers, start=1, xlabel='percent'):
        x_axis = list(range(start, 100))
        data = self._create_data(x_axis, headers)
        self._count_plot(x_axis, data, headers, xlabel)

    def class_plot(self, label):
        classes = sorted([int(c) for c in self[label].keys() if c != "--"])
        self._count_plot(
            x_axis=classes,
            data=self._create_data(classes, (label,)),
            xlabel=label,
            method="bar")


class Geotop(Plots):
    """
    Object with geotop counts.

    Running this for the first time will take time.
    When store_json is set to True and the file does not exist, it will save a
    json with the counts to disk (exploration/distributions/geotop.json).

    Has a plot method that shows a series of mapnik plots of the distributions.
    """
    json_filename = "geotop.json"
    dataset_names = RELEVANT_VARIABLES

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
    filenames = {
        'et': raster_filenames(root='et'),
        'rain': raster_filenames(root='rain'),
    }
    fraction = {'et': 10, 'rain': 0.01}

    def dataset_generator(self, key):
        for filepath in self.filenames[key]:
            yield (utils.get_h5_data(filepath, self.subdataset_name[key]) *
                   self.fraction[key]).astype('int')

    def plot(self, key):
        self.percent_plot((key, ), xlabel=key)

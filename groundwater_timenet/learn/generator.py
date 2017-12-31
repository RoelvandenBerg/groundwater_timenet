import os
from collections import Generator

import numpy as np

from groundwater_timenet import utils
from groundwater_timenet.learn.settings import *


class ConvolutionalAtrousGenerator(Generator):

    def __init__(self, base="neuralnet", data_type="train", batch_size=1000,
                 chunk_size=1000, meta_size=META_SIZE,
                 temporal_size=TEMPORAL_SIZE, input_size=INPUT_SIZE,
                 output_size=OUTPUT_SIZE):
        directory = os.path.join(utils.DATA, base, data_type)
        self.__length = None
        self.h5files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.h5')
        ]
        self.meta_size = meta_size
        self.temporal_size = temporal_size
        self.input_size = input_size
        self.output_size = output_size
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.input_data, self.output_data = self.empty_input_output()
        self.dataset_names = tuple(
            name + "_" + str(i) for name in ("base", "temporal", "meta")
            for i in range(self.chunk_size)
        )
        self.__generator = self.__generate()

    def send(self, _):
        return next(self.__generator)

    def __generate(self):
        # continue for ever:
        while True:
            for filepath in self.h5files:
                datasets = dict(self._datasets(filepath))
                for i in range(self.chunk_size):
                    base_data = datasets["base_" + str(i)]
                    skip = base_data != 0
                    condense_base = self.rolling_dataset(
                        dataset=base_data[skip],
                        period_length=self.output_size,
                        cutoff=self.input_size,
                        shift=self.input_size
                    )
                    base = np.zeros(
                        condense_base.shape[0] * self.input_size).reshape(
                        condense_base.shape[0], self.input_size, 1
                    )
                    base[:, self.input_size - 1, :] = condense_base
                    meta = datasets["meta_" + str(i)]
                    temporal = self.rolling_dataset(
                        dataset=datasets["temporal_" + str(i)][skip.flatten()],
                        period_length=self.input_size,
                        cutoff=self.output_size,
                    )
                    self.pack(base, meta, temporal)
                    for batch in self.unpack_batches():
                        yield batch

    def __len__(self):
        if self.__length is None:
            model_length = INPUT_SIZE + OUTPUT_SIZE
            self.__length = 0
            for f in self.h5files:
                datasets = self._datasets(f)
                self.__length += sum([
                    dataset.shape[0] - model_length for name, dataset in
                    datasets if name.startswith('base')
                ])
        return self.__length

    def empty_input_output(self):
        return (
            np.empty(
                shape=(0, self.input_size, self.temporal_size + self.meta_size)
            ),
            np.empty(shape=(0, self.input_size, 1))
        )

    def _datasets(self, filepath):
        return zip(
            self.dataset_names,
            utils.read_h5(filepath, dataset_name=self.dataset_names, many=True)
        )

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    @staticmethod
    def rolling_dataset(dataset, period_length, cutoff=0, shift=0):
        return np.array([
            np.roll(dataset, -i, axis=0)[shift:period_length + shift]
            for i in range(dataset.shape[0] - cutoff - period_length + 1)
        ])

    def pack(self, base, meta, temporal):
        self.output_data = np.concatenate([self.output_data, base])
        metadata = np.repeat(
            meta.reshape(1, -1), temporal.shape[0] * self.input_size, axis=0
        ).reshape(temporal.shape[0], self.input_size, self.meta_size)
        input_data = np.concatenate([temporal, metadata], axis=2)
        self.input_data = np.concatenate([self.input_data, input_data])

    def unpack_batches(self):
        batches_length = self.input_data.shape[0] % self.batch_size
        if batches_length == 0:
            inputs = self.input_data.copy()
            outputs = self.output_data.copy()
            self.input_data, self.output_data = self.empty_input_output()
        else:
            inputs = self.input_data[:-batches_length]
            outputs = self.output_data[:-batches_length]
            self.input_data = self.input_data[-batches_length:]
            self.output_data = self.output_data[-batches_length:]
        return zip(
            inputs.reshape(-1, self.batch_size, *self.input_data.shape[1:]),
            outputs.reshape(-1, self.batch_size, *self.output_data.shape[1:])
        )

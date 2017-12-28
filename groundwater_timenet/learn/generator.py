import datetime
import os
from collections import Generator

import numpy as np

from groundwater_timenet import utils
from groundwater_timenet.learn.constants import *

class ConvolutionalAtrousGenerator(Generator):

    def __init__(self, base="neuralnet", data_type="train", batch_size=1000,
                 chunk_size=1000, input_size=INPUT_SIZE,
                 output_size=OUTPUT_SIZE):
        directory = os.path.join(utils.DATA, base, data_type)
        self.h5files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.h5')
        ]
        self.before_size = before_size
        self.after_size = after_size
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.base_data = np.empty(shape=(0, 1))
        self.meta_data = np.empty(shape=(0, META_SIZE))
        self.temporal_data = np.empty(shape=(0, TEMPORAL_SIZE, before_size))
        self.dataset_name = tuple(
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
                datasets = dict(zip(
                    self.dataset_name,
                    utils.read_h5(
                        filepath, dataset_name=self.dataset_name, many=True
                    )
                ))
                for i in range(self.chunk_size):
                    base = self.rolling_dataset(
                        datasets["base_" + str(i)].reshape(1, -1),
                        self.after_size,
                        self.before_size,
                        1,
                    ).reshape(-1, 1)
                    meta = datasets["meta_" + str(i)]
                    temporal = self.rolling_dataset(
                            datasets["temporal_" + str(i)],
                            self.before_size,
                            self.after_size)
                    self.pack(base, meta, temporal)
                    for batch in self.unpack_batches():
                        yield batch

    def __len__(self):
        return self.base_data.shape[0]

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    @staticmethod
    def rolling_dataset(dataset, period_length, cutoff=0, shift=0):
        return np.array([
            np.roll(dataset, -i)[:, shift:period_length + shift] for i in
            range(dataset.shape[-1] - cutoff - period_length)
        ])

    def pack(self, base, meta, temporal):
        self.base_data = np.concatenate([self.base_data, base])
        self.meta_data = np.concatenate([
            self.meta_data,
            np.repeat(meta.reshape(1, -1), temporal.shape[0], axis=0)
        ])
        self.temporal_data = np.concatenate([self.temporal_data, temporal])

    def unpack_batches(self):
        batches_length = len(self) % self.batch_size
        batches = (
            self.temporal_data[:-batches_length].reshape(
                -1, self.batch_size, *self.temporal_data.shape[1:]),
            self.meta_data[:-batches_length].copy().reshape(
                -1, self.batch_size, *self.meta_data.shape[1:]),
            self.base_data[:-batches_length].copy().reshape(
                -1, self.batch_size, *self.base_data.shape[1:]),
        )
        self.temporal_data = self.temporal_data[batches_length:]
        self.meta_data = self.meta_data[-batches_length:]
        self.base_data = self.base_data[-batches_length:]
        return zip(*batches)

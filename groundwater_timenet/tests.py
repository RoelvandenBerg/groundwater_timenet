import unittest
import random

import numpy as np

from .parse.combine import Combiner, UncompressedCombiner
from .learn.generator import CompressedConvolutionalAtrousGenerator

# TODO: write more tests.


class CombinerTestCase(unittest.TestCase):

    def test_no_nans(self):
        # TODO: write a test that properly tests the NaN problem.
        c = Combiner()
        data_gen = c._base_data('train')
        for i in range(1):  # 150 is better, even better is to use fake data.
            x, y, z, start, end, base_metadata, base_data = next(data_gen)
            self.assertFalse(np.isnan(base_data).any())
            self.assertFalse(
                np.isnan(c.temporal_data(base_data, x, y, start, end)).any())
            self.assertFalse(
                np.isnan(c.meta_data(base_metadata, x, y, z)).any())

    def test_(self):
        c = UncompressedCombiner()
        end = random.randint(3, len(c._base_data))
        c._base_data._parts['testrun'] = slice(end - 3, end)
        for j, params in enumerate(c._base_data('testrun')):
            x, y, z, start, end, base_metadata, base = params
            temporal = c.temporal_data(base, x, y, start, end)
            meta = c.meta_data(base_metadata, x, y, z)
            self.assertEqual(temporal.shape[1], c.generator.temporal_size)
            self.assertEqual(base.shape[1], 1)
            self.assertEqual(meta.shape[0], c.generator.meta_size)
            self.assertEqual(temporal.shape[0], base.shape[0])
            self.assertEqual(len(temporal.shape), 2)
            self.assertEqual(len(base.shape), 2)
            self.assertEqual(len(meta.shape), 1)


class GeneratorTestCase(unittest.TestCase):

    expected_input_data = np.array(
        [[[100.,  101.,    0.,    1.,    2.],
          [102.,  103.,    0.,    1.,    2.],
          [104.,  105.,    0.,    1.,    2.],
          [106.,  107.,    0.,    1.,    2.],
          [108.,  109.,    0.,    1.,    2.]],

         [[110.,  111.,    0.,    1.,    2.],
          [112.,  113.,    0.,    1.,    2.],
          [114.,  115.,    0.,    1.,    2.],
          [116.,  117.,    0.,    1.,    2.],
          [118.,  119.,    0.,    1.,    2.]],

         [[120.,  121.,    0.,    1.,    2.],
          [122.,  123.,    0.,    1.,    2.],
          [124.,  125.,    0.,    1.,    2.],
          [126.,  127.,    0.,    1.,    2.],
          [128.,  129.,    0.,    1.,    2.]],

         [[130.,  131.,    0.,    1.,    2.],
          [132.,  133.,    0.,    1.,    2.],
          [134.,  135.,    0.,    1.,    2.],
          [136.,  137.,    0.,    1.,    2.],
          [138.,  139.,    0.,    1.,    2.]],

         [[140.,  141.,    0.,    1.,    2.],
          [142.,  143.,    0.,    1.,    2.],
          [144.,  145.,    0.,    1.,    2.],
          [146.,  147.,    0.,    1.,    2.],
          [148.,  149.,    0.,    1.,    2.]],

         [[150.,  151.,    0.,    1.,    2.],
          [152.,  153.,    0.,    1.,    2.],
          [154.,  155.,    0.,    1.,    2.],
          [156.,  157.,    0.,    1.,    2.],
          [158.,  159.,    0.,    1.,    2.]],

         [[160.,  161.,    0.,    1.,    2.],
          [162.,  163.,    0.,    1.,    2.],
          [164.,  165.,    0.,    1.,    2.],
          [166.,  167.,    0.,    1.,    2.],
          [168.,  169.,    0.,    1.,    2.]],

         [[170.,  171.,    0.,    1.,    2.],
          [172.,  173.,    0.,    1.,    2.],
          [174.,  175.,    0.,    1.,    2.],
          [176.,  177.,    0.,    1.,    2.],
          [178.,  179.,    0.,    1.,    2.]],

         [[180.,  181.,    0.,    1.,    2.],
          [182.,  183.,    0.,    1.,    2.],
          [184.,  185.,    0.,    1.,    2.],
          [186.,  187.,    0.,    1.,    2.],
          [188.,  189.,    0.,    1.,    2.]],

         [[190.,  191.,    0.,    1.,    2.],
          [192.,  193.,    0.,    1.,    2.],
          [194.,  195.,    0.,    1.,    2.],
          [196.,  197.,    0.,    1.,    2.],
          [198.,  199.,    0.,    1.,    2.]]]
    )

    def setUp(self):
        self.gen = CompressedConvolutionalAtrousGenerator(
            input_size=5,
            output_size=1,
            temporal_size=2,
            meta_size=3,
            batch_size=2,
            chunk_size=4
        )
        self.temporal = (np.arange(2 * 5 * 10) + 100).reshape(10, 5, 2)
        self.base = np.zeros(5 * 10).reshape(10, 5, 1)
        self.base[:, 4, :] = np.arange(10).reshape(10, 1)
        self.meta = np.arange(3)

    def test_rolling_dataset(self):
        expected_roll = np.array([
            [[100, 101],
             [102, 103]],

            [[102, 103],
             [104, 105]],

            [[104, 105],
             [106, 107]],

            [[106, 107],
             [108, 109]],
        ])

        rolled_1 = self.gen.rolling_dataset(self.temporal[0], 2, cutoff=1)
        rolled_2 = self.gen.rolling_dataset(
            self.temporal[0], 2, cutoff=1, shift=1)
        self.assertEqual((3, 2, 2), rolled_1.shape)
        self.assertEqual((3, 2, 2), rolled_2.shape)
        self.assertTrue((expected_roll[:-1] == rolled_1).all())
        self.assertTrue((expected_roll[1:] == rolled_2).all())

    def test_pack(self):
        self.gen.pack(self.base, self.meta, self.temporal)
        self.assertTrue((self.base == self.gen.output_data).all())
        self.assertTrue(
            (self.expected_input_data == self.gen.input_data).all())

    def test_unpack(self):
        self.gen.input_data = self.expected_input_data
        self.gen.output_data = self.base
        input, output = next(self.gen.unpack_batches())
        expected_input = np.array(
            [[[100.,  101.,    0.,    1.,    2.],
              [102.,  103.,    0.,    1.,    2.],
              [104.,  105.,    0.,    1.,    2.],
              [106.,  107.,    0.,    1.,    2.],
              [108.,  109.,    0.,    1.,    2.]],

             [[110.,  111.,    0.,    1.,    2.],
              [112.,  113.,    0.,    1.,    2.],
              [114.,  115.,    0.,    1.,    2.],
              [116.,  117.,    0.,    1.,    2.],
              [118.,  119.,    0.,    1.,    2.]]]
        )
        expected_output = np.array([
            [[0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [1]]
        ])
        self.assertTrue((expected_input == input).all())
        print(expected_output, output)
        print(expected_output.shape, output.shape)
        self.assertTrue((expected_output == output).all())


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from groundwater_timenet.parse.combine import Combiner

# TODO: write real tests.


class CombinerTestCase(unittest.TestCase):

    def test_no_nans(self):
        # TODO: write a test that properly tests the NaN problem.
        c = Combiner()
        data_gen = c._base_data('train')
        for i in range(150):
            x, y, z, start, end, base_metadata, base_data = next(data_gen)
            self.assertFalse(np.isnan(base_data).any())
            self.assertFalse(
                np.isnan(c.temporal_data(base_data, x, y, start, end)).any())
            self.assertFalse(
                np.isnan(c.meta_data(base_metadata, x, y, z)).any())


if __name__ == '__main__':
    unittest.main()

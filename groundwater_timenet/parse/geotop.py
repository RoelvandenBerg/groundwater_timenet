import os

from netCDF4 import Dataset
import numpy as np

from groundwater_timenet import utils
from groundwater_timenet.parse.base import Data


logger = utils.setup_logging(__name__, utils.PARSE_LOG)

RELEVANT_VARIABLES = (
    'strat',
    # 'lithok',  # lithok is a combination of the chance variables below
    'kans_1',
    'kans_2',
    'kans_3',
    'kans_4',
    'kans_5',
    'kans_6',
    'kans_7',
    'kans_8',
    'kans_9',
    'onz_lk',
    'onz_ls'
)

STRAT_CLASSES = {
    '--': 0, '1000': 1, '1010': 2, '1020': 3, '1030': 4, '1040': 5, '1045': 6,
    '1050': 7, '1070': 8, '1080': 9, '1090': 10, '1100': 11, '1110': 12,
    '1120': 13, '1130': 14, '2000': 15, '2010': 16, '3000': 17, '3011': 18,
    '3012': 19, '3020': 20, '3030': 21, '3050': 22, '3100': 23, '4000': 24,
    '4010': 25, '4080': 26, '4100': 27, '4110': 28, '5000': 29, '5010': 30,
    '5020': 31, '5030': 32, '5040': 33, '5050': 34, '5060': 35, '5070': 36,
    '5080': 37, '5090': 38, '5120': 39, '5130': 40, '5140': 41, '5150': 42,
    '5180': 43, '5200': 44, '5230': 45, '5260': 46, '6000': 47, '6010': 48,
    '6100': 49, '6110': 50, '6200': 51, '6300': 52, '6320': 53, '6400': 54,
    '6420': 55
}


class GeotopData(Data):
    root = "geotop"
    type = Data.DataType.METADATA

    def __init__(self, filename='geotop.nc', *args, **kwargs):
        super(GeotopData, self).__init__(*args, **kwargs)
        self.filepath = os.path.join(utils.DATA, self.root, filename)
        self.rootgrp = Dataset(self.filepath, "r")

    def _data(self, x, y, z=0, *args, **kwargs):
        depth = int(round((z + 50) * 2))
        rd_x = int(round(x - 13600) / 100)
        rd_y = int(round(y - 358000) / 100)
        return [
            self.rootgrp[variable][rd_x, rd_y, depth]
            for variable in RELEVANT_VARIABLES
        ]

    def _strat(self, code):
        zeros = np.zeros(56)
        zeros[STRAT_CLASSES[str(code)]] = 1
        return zeros

    def _normalize(self, data):
        return np.concatenate(
            [self._strat(data[0]), np.array(data[1:]) / 100.0])


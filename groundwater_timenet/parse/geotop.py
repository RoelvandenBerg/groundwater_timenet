from netCDF4 import Dataset
import os

try:
    from groundwater_timenet import utils
except ImportError:
    from .. import utils


logger = utils.setup_logging(__name__, utils.PARSE_LOG)

RELEVANT_VARIABLES = [
    'strat',
    'lithok',
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
]


def geotop_handler(filename='geotop.nc'):
    filepath = os.path.join(utils.DATA, 'geotop', filename)
    rootgrp = Dataset(filepath, "r")

    def geotop_data(x, y, ground_level):
        z = int(round((ground_level + 50) * 2))
        rd_x = int(round(x - 13600) / 100)
        rd_y = int(round(y - 358000) / 100)
        return [
            rootgrp[variable][rd_x, rd_y, z]
            for variable in RELEVANT_VARIABLES
        ]

    return geotop_data

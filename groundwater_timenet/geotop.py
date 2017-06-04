from netCDF4 import Dataset
from urllib.request import urlopen
import os

try:
    from groundwater_timenet import utils
except ImportError:
    import utils


logger = utils.setup_logging(__name__, utils.HARVEST_LOG)

GEOTOP_URL = "http://www.dinodata.nl/opendap/GeoTOP/geotop.nc"
CHUNK = 16 * 1024

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


def download_large_file(url, filepath):
    """
    Stackoverflow is your friend:
    https://stackoverflow.com/questions/1517616/
        stream-large-binary-files-with-urllib2-to-file#answer-1517728
    Kudos to Alex Martelli.
    """
    response = urlopen(url)
    with open(filepath, 'wb') as f:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)


def geotop_handler(filename='geotop.nc'):
    filepath = os.path.join(utils.DATA, 'geotop', filename)
    rootgrp = Dataset(filepath, "r")

    def geotop_data(x, y, ground_level):
        z = int(round((ground_level + 50) * 2))
        rd_x = int(round(x - 13600) / 100)
        rd_y = int(round(x - 358000) / 100)
        return [
            rootgrp[variable][rd_x, rd_y, z]
            for variable in RELEVANT_VARIABLES
        ]

    return geotop_data


def download(filename='geotop.nc'):
    filepath = os.path.join(utils.DATA, 'geotop', filename)
    download_large_file(GEOTOP_URL, filepath)

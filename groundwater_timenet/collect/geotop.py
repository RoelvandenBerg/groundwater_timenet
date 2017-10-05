from urllib.request import urlopen
import os

from groundwater_timenet import utils


logger = utils.setup_logging(__name__, utils.HARVEST_LOG)
GEOTOP_URL = "http://www.dinodata.nl/opendap/GeoTOP/geotop.nc"
CHUNK = 16 * 1024


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


def download(filename='geotop.nc'):
    filepath = os.path.join(utils.DATA, 'geotop', filename)
    download_large_file(GEOTOP_URL, filepath)

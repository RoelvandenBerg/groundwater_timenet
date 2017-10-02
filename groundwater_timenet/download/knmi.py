import io
import zipfile

import requests

try:
    from groundwater_timenet import utils
except ImportError:
    from .. import utils


logger = utils.setup_logging(__name__, utils.HARVEST_LOG)

STATION_URL = ("https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/"
               "daggegevens/etmgeg_{code}.zip")

STATION_CODES = [
    '210', '215', '235', '240', '249', '251', '257', '260', '265', '267',
    '269', '270', '273', '275', '277', '278', '279', '280', '283', '286',
    '290', '310', '319', '323', '330', '344', '348', '350', '356', '370',
    '375', '377', '380', '391'
]


def download_measurementstation_metadata():
    """
    Downloads a file with in the header
    :return:
    """
    data = {'stns': "ALL"}
    url = "http://projects.knmi.nl/klimatologie/monv/reeksen/getdata_rr.cgi"
    response = requests.post(url=url, data=data)
    with open('var/data/knmi_measurementstations/knmi_metadata.txt', 'w'
              ) as f:
        f.write(response.text)


def load_knmi_measurement_data(target_dir='var/data/knmi_measurementstations'):
    """Downloads all knmi measurementstation data to target_dir."""
    urls = ((code, STATION_URL.format(code=code)) for code in STATION_CODES)
    utils.mkdirs(target_dir)
    for code, url in urls:
        response = requests.get(url)
        zf = zipfile.ZipFile(io.BytesIO(response.content))
        zf.extractall(path=target_dir)
        logger.debug("Collected measurement data for station %s", code)



if __name__ == '__main__':
    logger.info('These grids can more easily be downloaded from this '
                'location for rain: '
                'https://data.knmi.nl/datasets/radar_corr_accum_24h/1.0 '
                'and this location for evaporation:'
                'https://data.knmi.nl/datasets/EV24/2')
    load_knmi_measurement_data()


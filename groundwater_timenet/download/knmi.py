import io
import os
import re
import zipfile

import requests
import numpy as np

from groundwater_timenet import utils


logger = utils.setup_logging(__name__, utils.HARVEST_LOG)

STATION_URL = ("https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/"
               "daggegevens/etmgeg_{code}.zip")

STATION_CODES = [
    '210', '215', '235', '240', '249', '251', '257', '260', '265', '267',
    '269', '270', '273', '275', '277', '278', '279', '280', '283', '286',
    '290', '310', '319', '323', '330', '344', '348', '350', '356', '370',
    '375', '377', '380', '391'
]

HEADER = (
    '# STN,YYYYMMDD,DDVEC,FHVEC,   FG,  FHX, FHXH,  FHN, FHNH,  FXX, '
    'FXXH,   TG,   TN,  TNH,   TX,  TXH, T10N,T10NH,   SQ,   SP,    Q,   '
    'DR,   RH,  RHX, RHXH,   PG,   PX,  PXH,   PN,  PNH,  VVN, VVNH,  '
    'VVX, VVXH,   NG,   UG,   UX,  UXH,   UN,  UNH, EV24\n'
)
EXAMPLE_ET_PATH = (
    'var/data/et/2010/10/01/'
    'INTER_OPER_R___EV24____L3__20101001T000000_20101002T000000_0002.nc'
)
EXAMPLE_RAIN_PATH = "var/data/rain/2010/10/01/RAD_NL25_RAC_24H_201010020800.h5"


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


def _measurement_stations(directory):
    for filename in os.listdir(directory):
        if 'etmgeg' in filename:
            id_ = filename.replace('etmgeg_', '').replace('.txt', '')
            path = os.path.join(directory, filename)
            with open(path, 'r') as f:
                data = f.read().split(HEADER)[1]
                data_lines = data.split('\n')
                yield (id_, [
                    [utils.int_or_nan(l) for l in line.split(',')]
                    for line in data_lines
                    if line
                    ])


def cache_measurement_station_data(
        directory='var/data/knmi_measurementstations'):
    station_data = dict(_measurement_stations(directory))
    utils.store_h5(
        data=[np.array(d) for d in station_data.values()],
        dataset_name=[str(k) for k in station_data.keys()],
        target_h5="var/data/knmi/measurementstations.h5",
        many=True
    )


def reshape_rasters(root, grid_size=50):
    dataset_name, shape, slice_method = {
        "et": ('prediction', (350, 300), lambda si, sj: (0, si, sj)),
        'rain': ('image1/image_data', (765, 700), lambda si, sj: (si, sj))
    }[root]
    files = utils.raster_filenames(root, raise_errors=False)
    regex = re.compile(r'_(\d{4})(\d{2})(\d{2})', re.UNICODE)
    timestamps = np.array([regex.findall(f)[0] for f in files]).astype('int')
    for i in range(0, shape[0], grid_size):
        for j in range(0, shape[1], grid_size):
            data = np.array(
                [
                    utils.read_h5(
                        filepath=f,
                        dataset_name=dataset_name,
                        index=slice_method(slice(i, i + grid_size),
                                           slice(j, j + grid_size)))
                    for f in files
                ]
            )
            target = "var/data/knmi/{root}/{i}/{j}.h5".format(
                root=root, i=i, j=j)
            utils.mkdirs(target)
            utils.store_h5(
                data=[np.moveaxis(data, 0, -1), timestamps],
                dataset_name=["data", "timestamps"],
                target_h5=target,
                many=True
            )


if __name__ == '__main__':
    logger.info('These grids can more easily be downloaded from this '
                'location for rain: '
                'https://data.knmi.nl/datasets/radar_corr_accum_24h/1.0 '
                'and this location for evaporation:'
                'https://data.knmi.nl/datasets/EV24/2')
    load_knmi_measurement_data()
    cache_measurement_station_data()
    if not os.path.exists(EXAMPLE_RAIN_PATH) and not os.path.exists(
            EXAMPLE_ET_PATH):
        reshape_rasters('rain')
        reshape_rasters('et')
    else:
        raise OSError(
            'Data is missing please download the rain and evaporation data. '
            'Rain: https://data.knmi.nl/datasets/radar_corr_accum_24h/1.0 '
            'Evaporation: https://data.knmi.nl/datasets/EV24/2'
        )

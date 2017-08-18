import datetime
import ftplib
import io
import os
import zipfile

import osr
import gdal
import h5py
import numpy as np
import requests

try:
    from groundwater_timenet import utils
except ImportError:
    from .. import utils


logger = utils.setup_logging(__name__, utils.HARVEST_LOG)

FILENAME_BASE = "knmi"
RAIN_NAN_VALUE = 65535
ET_NAN_VALUE = -9999.0

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


def grab_daily_ftp(target_dir, source_base, filename_parser, start, end=None,
                   ftp_base='data.knmi.nl'):
    """
    Grab daily files through ftp.

    :param target_dir: directory to put the files
    :param source_base: ftp base directory where the files can be found
        (followed by /year/month/day/ directorystructure).
    :param filename_parser: function that parses a filename from a date
    :param start: first date tuple (year, month, day)
    :param end: last datetime (defaults to today)
    :param ftp_base: ftp url
    """
    start = datetime.datetime(*start)
    if not end:
        end = datetime.datetime.combine(
            datetime.date.today(), datetime.time(8))

    period_days = (end - start).days
    dates = (start + datetime.timedelta(days=x) for x in range(period_days))

    # connect to domain name:
    ftp = ftplib.FTP(ftp_base)
    ftp.login()

    # iterate over filenames
    for date in dates:
        filename = filename_parser(date)
        date_dir = os.path.join(
            str(date.year),
            ("0" + str(date.month))[-2:]
        )
        target_file_path = os.path.join(target_dir, date_dir, filename)
        # ensure directory exists, otherwise create it:
        utils.mkdirs(target_file_path)

        source_dir = os.path.join(
            source_base,
            date_dir,
            ("0" + str(date.day))[-2:]
        )
        logger.info("Downloading file to %s", target_file_path)
        # change to relevant folder
        ftp.cwd(source_dir)
        # fetch and write file
        with open(target_file_path, 'wb') as localfile:
            ftp.retrbinary('RETR ' + filename, localfile.write, 1024)
    ftp.quit()


def grab_rain_grids(target_dir="var/data/rain/", start=(2008, 3, 11, 8)):
    """Downloads all knmi evapotranspiration data to target_dir."""
    day = datetime.timedelta(1)
    change_date = datetime.datetime(2010, 5, 31)

    def parser(date):
        if date >= change_date:
            return "RAD_NL25_RAC_24H_" + (date + day
                                          ).strftime("%Y%m%d%H%M") + ".h5"
        else:
            return "RAD_NL25_RAC_24H_" + date.strftime("%Y%m%d%H%M") + ".h5"
    grab_daily_ftp(
        target_dir=target_dir,
        source_base='/download/radar_corr_accum_24h/1.0/noversion/',
        filename_parser=parser,
        start=start
    )


def grab_evap_grids(target_dir="var/data/et/"):
    """Downloads all knmi evapotranspiration data to target_dir."""
    day = datetime.timedelta(1)

    def parse(x):
        return x.date().isoformat().replace("-", "") + "T000000_"

    def fn_parser(date):
        return "INTER_OPER_R___EV24____L3__" + parse(date) + \
               parse(date + day) + "0002.nc"

    grab_daily_ftp(
        target_dir=target_dir,
        source_base="/download/EV24/2/0002/",
        filename_parser=fn_parser,
        start=(1965, 1, 1)
    )


def load_knmi_measurement_data(target_dir='var/data/knmi_measurementstations'):
    """Downloads all knmi measurementstation data to target_dir."""
    urls = ((code, STATION_URL.format(code=code)) for code in STATION_CODES)
    utils.mkdirs(target_dir)
    for code, url in urls:
        response = requests.get(url)
        zf = zipfile.ZipFile(io.BytesIO(response.content))
        zf.extractall(path=target_dir)
        logger.debug("Collected measurement data for station %s", code)


def try_h5(f):
    try:
        h5py.File(f, 'r')
    except OSError:
        return f


def get_raster_filenames(rootdir):
    files = sorted(
        [os.path.join(r, f[0]) for r, d, f in os.walk('var/data/' + rootdir) if f])
    faulty = [x for x in [try_h5(f) for f in files] if x]
    if faulty:
        raise OSError(
            'Broken HDF5 files found:\n- {}'.format('\n- '.join(faulty)))
    return files


def rain_timeseries(x, y, files):
    for filepath in files:
        h5file = h5py.File(filepath, 'r', libver='latest')
        dataset = h5file.get('image1/image_data')
        try:
            value = dataset[x,y]
        except:
            logger.exception(
                'file %s does not have dataset image1/image_data', filepath)
            yield np.NaN
        if value == RAIN_NAN_VALUE:
            yield np.NaN
        else:
            yield value


def rain_timeseries_dataset(minx, miny, maxx, maxy, files, _=None):
    #TODO: bug
    print(minx, miny, maxx, maxy)
    # this should return a 10x10 matrix for the 10.000 m gridcell
    return np.array([
        [
            list(rain_timeseries(x, y, files))
            for y in range(miny, maxy)
        ] for x in range(minx, maxx)
    ])


def points_list(example_file):
    h5file = h5py.File(example_file, 'r', libver='latest')
    lat = h5file.get('lat')
    lon = h5file.get('lon')
    val = [
        (p, x, y)
        for p, x, y in [
            (utils.point(float(lon[x, y]), float(lat[x, y])), x, y)
            for x in range(350)
            for y in range(300)
        ] if not utils.transform(p)
    ]
    return val


def et_timeseries(x, y, files):
    for filepath in files:
        h5file = h5py.File(filepath, 'r', libver='latest')
        dataset = h5file.get('prediction')
        value = dataset[0, x, y]
        if value == ET_NAN_VALUE:
            yield np.NaN
        else:
            yield value


def et_timeseries_dataset(minx, miny, maxx, maxy, files, points):
    grid = [
        (x, y) for point, x, y in points
        if utils.within(point, minx, miny, maxx, maxy)
    ]
    # this should result in one value per 10.000m gridcell
    return np.array([
        [
            list(et_timeseries(x, y, files))
            for x, y in grid
        ]
    ])


def add_timeseries_data(minx, miny, maxx, maxy, files,
                        timeseries_dataset_method, h5_file, dataset_name,
                        points):
    data = timeseries_dataset_method(minx, miny, maxx, maxy, files, points)
    dataset = h5_file.create_dataset(
        dataset_name,
        data.shape,
        dtype=data.dtype)
    dataset[...] = data


def apply_transform(x, y, coord_transform,
                    asine=(0.0, 1.0, 0, -3649.98, 0, -1.0)):
    return [round(f) for f in gdal.ApplyGeoTransform(
        asine, *coord_transform.TransformPoint(x, y)[:2])]


def reshape_rasters(rain_root='rain', et_root='et'):
    rain_files = get_raster_filenames(rain_root)
    et_files = get_raster_filenames(et_root)
    et_points = points_list(et_files[0])

    rain_proj = osr.SpatialReference(osr.GetUserInputAsWKT(
        '+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378.14 +b=6356.75 '
        '+x_0=0 y_0=0'))
    rd_proj = osr.SpatialReference(osr.GetUserInputAsWKT('epsg:28992'))
    sliding_geom_window = utils.sliding_geom_window('NederlandRegion.json')
    coord_transform = osr.CoordinateTransformation(rd_proj, rain_proj)
    for minx, miny, maxx, maxy in sliding_geom_window:
        filepath = utils.parse_filepath(
            minx, miny, filename_base=FILENAME_BASE)
        h5_file = h5py.File(filepath, "w", libver='latest')
        rain_minx, rain_miny = apply_transform(minx, miny, coord_transform)
        rain_maxx, rain_maxy = apply_transform(maxx, maxy, coord_transform)
        add_timeseries_data(
            rain_minx, rain_miny, rain_maxx, rain_maxy, rain_files,
            rain_timeseries_dataset, h5_file, rain_root, None
        )
        add_timeseries_data(
            minx, miny, maxx, maxy, et_files, et_timeseries_dataset,
            h5_file, et_root, et_points
        )
        logger.debug("Added rain and et data for minx %d, miny %d, maxx %d, maxy %d", minx, miny, maxx, maxy)


if __name__ == '__main__':
    logger.info('These grids can more easily be downloaded from this '
                'location for rain: '
                'https://data.knmi.nl/datasets/radar_corr_accum_24h/1.0 '
                'and this location for evaporation:'
                'https://data.knmi.nl/datasets/EV24/2')
    # grab_evap_grids()
    # grab_rain_grids()
    load_knmi_measurement_data()


# # in case you used the grab_..._grids methods, you need to put the daily files
# # each in its own directory, this is for the et-grids:
# def digitify(digit):
#     return '0' * (2 - len(str(digit))) + str(digit)
#
#
# for year in range(1965, 2018):
#      for month in range(1, 13):
#          month_root = os.path.join(root, str(year), digitify(month))
#          files = os.listdir(month_root)
#          for i, daydir in enumerate(range(1, len(files) + 1)):
#              new_dir_path = os.path.join(month_root, digitify(daydir))
#              os.makedirs(new_dir_path)
#              filepath = os.path.join(month_root, files[i])
#              new_filepath = os.path.join(new_dir_path, files[i])
#              os.rename(filepath, new_filepath)
#              print('moved', filepath, 'to', new_filepath)
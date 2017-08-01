import datetime
import ftplib
import io
import os
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
    '209', '210', '215', '225', '235', '240', '242', '248', '249', '251',
    '257', '258', '260', '265', '267', '269', '270', '273', '275', '277',
    '278', '279', '280', '283', '285', '286', '290', '308', '310', '311',
    '312', '313', '315', '316', '319', '323', '324', '330', '331', '340',
    '343', '344', '348', '350', '356', '370', '375', '377', '380', '391'
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


if __name__ == '__main__':
    grab_evap_grids()
    grab_rain_grids()
    load_knmi_measurement_data()

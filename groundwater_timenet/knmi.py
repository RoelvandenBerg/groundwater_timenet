import zipfile
import ftplib
import os
import gzip
import datetime

import requests


def grab_files(target_dir, source_base, filename_parser, start, end=None):
    """
    
    :param data_dir: 
    :param first_time: 
    :param last_year: 
    :return: 
    """
    start = datetime.datetime(*start)
    if not end:
        end = datetime.datetime.combine(datetime.date.today(), datetime.time(8))

    period_days = (end - start).days
    dates = (start + datetime.timedelta(days=x) for x in range(period_days))

    # connect to domain name:
    ftp = ftplib.FTP('data.knmi.nl/')
    ftp.login()

    # iterate over filenames
    for date in dates:
        filename = filename_parser(date)
        file_path = os.path.join(target_dir, filename)
        source_dir = os.path.join(
            source_base,
            ("0" + str(date.month))[-2:],
            str(date.year)
        )
        # change to relevant folder
        ftp.cwd(source_dir)
        # fetch and write file
        with open(file_path, 'wb') as localfile:
            ftp.retrbinary('RETR ' + filename, localfile.write, 1024)
    ftp.quit()


def grab_rain_grids():
    grab_files(
        target_dir="var/data/rain/",
        source_base='/download/radar_corr_accum_24h/1.0/noversion/',
        filename_parser=lambda date: "RAD_NL25_RAC_24H_" + date.strftime("%Y%m%d%H%M") + ".h5",
        start=(2008, 3, 11, 8)
    )


def grab_evap_grids():
    day = datetime.timedelta(1)
    def fn_parser(date):
        parse = lambda x: x.date().isoformat().replace("-", "") + "T0000_"
        "INTER_OPER_R___EV24____L3__" + parse(date) + parse(date + day) + "0002.nc"
    grab_files(
        target_dir="var/data/et/",
        source_base="/download/EV24/2/0002/",
        filename_parser=fn_parser,
        start=(1965, 1, 1)
    )


def load_knmi_rain():
    pass
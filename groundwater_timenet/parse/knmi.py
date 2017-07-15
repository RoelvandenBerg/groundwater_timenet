import logging
import os
import random

import h5py
import numpy as np

try:
    from groundwater_timenet import utils
    from groundwater_timenet.download import knmi as download
except ImportError:
    from .. import utils
    from ..download import knmi as download


logger = utils.setup_logging(__name__, utils.PARSE_LOG, logging.INFO)

HEADER = (
    '# STN,YYYYMMDD,DDVEC,FHVEC,   FG,  FHX, FHXH,  FHN, FHNH,  FXX, FXXH,   '
    'TG,   TN,  TNH,   TX,  TXH, T10N,T10NH,   SQ,   SP,    Q,   DR,   RH,  '
    'RHX, RHXH,   PG,   PX,  PXH,   PN,  PNH,  VVN, VVNH,  VVX, VVXH,   NG,   '
    'UG,   UX,  UXH,   UN,  UNH, EV24\n'
)

def raw_data(directory):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        with open(path, 'r') as f:
            data = f.read().split(HEADER)[1]
            data_lines = data.split('\n')
            yield [
                [int(l.strip(" ")) for l in line.split(',')]
                for line in data_lines
            ]

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

HISTORICAL_KNMI_LOCATIONS = {
    ("210", "Valkenburg"): (88625, 466577),
    ("235", "De Kooy" ): (114397, 549755),
    ("240", "Schiphol"): (113824, 481139),
    ("242", "Vlieland"): (123580, 583073),
    ("249", "Berkhout"): (127668, 518133),
    ("251", "Hoorn Terschelling"): (152524, 599665),
    ("257", "Wijk aan Zee"): (101546, 501657),
    ("260", "De Bilt"): (141031, 456881),
    ("265", "Soesterberg"): (147888, 460575),
    ("267", "Stavoren"): (154739, 545876),
    ("269", "Lelystad"): (163801, 495811),
    ("270", "Leeuwarden"): (179234, 581177),
    ("273", "Marknesse"): (188536, 523736),
    ("275", "Deelen"): (187889, 451407),
    ("277", "Lauwersoog"): (209042, 603680),
    ("278", "Heino"): (214809, 494312),
    ("279", "Hoogeveen"): (234635, 529835),
    ("280", "Eelde"): (235084, 570652),
    ("283", "Hupsel"): (241587, 453906),
    ("286", "Nieuw Beerta"): (272792, 580703),
    ("290", "Twenthe"): (257125, 476459),
    ("310", "Vlissingen"): (30775, 386068),
    ("319", "Westdorpe"): (48767, 359693),
    ("323", "Wilhelminadorp"): (50658, 394893),
    ("330", "Hoek van Holland"): (67720, 444644),
    ("340", "Woensdrecht"): (82903, 385058),
    ("344", "Rotterdam"): (90594, 442442),
    ("348", "Cabauw"): (122663, 442132),
    ("350", "Gilze-Rijen"): (123531, 397623),
    ("356", "Herwijnen"): (138656, 429074),
    ("370", "Eindhoven"): (154731, 384546),
    ("375", "Volkel"): (176648, 406843),
    ("377", "Ell"): (181522, 356802),
    ("380", "Maastricht"): (181696, 323428),
    ("391", "Arcen"): (211437, 390423),

    # Antique stations:
    ("049", "Delft, (Leiden) en Rijnsburg"): (83775, 446244),
    ("056", "Breda"): (113146, 399553),
    ("057", "Breda"): (113146, 399553),
    ("043", "Utrecht"): (135321, 456900),
    ("048", "Haarlem"): (104809, 488641),
    ("044", "Haarlem"): (104809, 488641),
    ("042", "Zwanenburg"): (110482, 488586),
    ("050", "Leiden"): (94301, 464650),
    ("047", "Bergen"): (109639, 520122),
    ("046", "Alkmaar"): (111878, 518247),
    ("052", "Amsterdam"): (121817, 486643),
    ("053", "Amsterdam"): (121817, 486643),
    ("054", "Vlissingen"): (28458, 386126),
    ("003", "Vlissingen"): (28458, 386126),
    ("045", "Delft"): (84919, 446228),
    ("055", "Middelburg en Oostkapelle"): ((32068, 391601), (25317, 399190)),
    ("001", "De Bilt"): (141031, 456881),
    ("002", "Den Helder"): (112189, 553484),
    ("006", "Groningen"): (232672, 581743),
    ("007", "Maastricht"): (175856, 317838),
    # Diverse Antieke reeksen
    ("", "Leiden (Senguerdius)"): (94279, 462795),
    ("", "Fremery Utrecht"): (138747, 456888),
    # KNMI-neerslagstations
    ("009", "Den Helder"): (112189, 553484),
    ("011", "West Terschelling"): (143647, 597823),
    ("025", "De Kooy"): (114366, 546046),
    ("139", "Groningen"): (236043, 579944),
    ("144", "Ter Apel"): (268045, 545336),
    ("222", "Hoorn"): (133308, 518105),
    ("328", "Heerde"): (200108, 490446),
    ("438", "Hoofddorp"): (105833, 477503),
    ("550", "De Bilt"): (139883, 455030),
    ("666", "Winterswijk"): (245182, 444696),
    ("737", "Kerkwerve"): (49850, 411603),
    ("745", "Axel"): (51209, 365207),
    ("770", "Westdorpe"): (47603, 359718),
    ("828", "Oudenbosch"): (95798, 397872),
    ("961", "Roermond"): (195516, 355040),
}


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

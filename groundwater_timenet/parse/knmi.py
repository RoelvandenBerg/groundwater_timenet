import os

try:
    from groundwater_timenet import utils
    from groundwater_timenet.download import knmi as download
except ImportError:
    from .. import utils
    from ..download import knmi as download


logger = utils.setup_logging(__name__, utils.PARSE_LOG, "INFO")


def int_or_none(x):
    try:
        return int(x)
    except ValueError:
        return None


class WeatherStations(object):
    HEADER = (
        '# STN,YYYYMMDD,DDVEC,FHVEC,   FG,  FHX, FHXH,  FHN, FHNH,  FXX, '
        'FXXH,   TG,   TN,  TNH,   TX,  TXH, T10N,T10NH,   SQ,   SP,    Q,   '
        'DR,   RH,  RHX, RHXH,   PG,   PX,  PXH,   PN,  PNH,  VVN, VVNH,  '
        'VVX, VVXH,   NG,   UG,   UX,  UXH,   UN,  UNH, EV24\n'
    )

    # All uncommented stations contain both rain and evaporation
    STATION_META = [
        #     ("209", (4.518, 52.465, 0.00, "IJMOND")),
        ("210", (4.430, 52.171, -0.20, "VALKENBURG")),
        ("215", (4.437, 52.141, -1.10, "VOORSCHOTEN")),
        #     ("225", (4.555, 52.463, 4.40, "IJMUIDEN")),
        ("235", (4.781, 52.928, 1.20, "DE KOOY")),
        ("240", (4.790, 52.318, -3.30, "SCHIPHOL")),
        #     ("242", (4.921, 53.241, 10.80, "VLIELAND")),
        #     ("248", (5.174, 52.634, 0.80, "WIJDENES")),
        ("249", (4.979, 52.644, -2.40, "BERKHOUT")),
        ("251", (5.346, 53.392, 0.70, "HOORN (TERSCHELLING)")),
        ("257", (4.603, 52.506, 8.50, "WIJK AAN ZEE")),
        #     ("258", (5.401, 52.649, 7.30, "HOUTRIBDIJK")),
        ("260", (5.180, 52.100, 1.90, "DE BILT")),
        ("265", (5.274, 52.130, 13.90, "SOESTERBERG")),
        ("267", (5.384, 52.898, -1.30, "STAVOREN")),
        ("269", (5.520, 52.458, -3.70, "LELYSTAD")),
        ("270", (5.752, 53.224, 1.20, "LEEUWARDEN")),
        ("273", (5.888, 52.703, -3.30, "MARKNESSE")),
        ("275", (5.873, 52.056, 48.20, "DEELEN")),
        ("277", (6.200, 53.413, 2.90, "LAUWERSOOG")),
        ("278", (6.259, 52.435, 3.60, "HEINO")),
        ("279", (6.574, 52.750, 15.80, "HOOGEVEEN")),
        ("280", (6.585, 53.125, 5.20, "EELDE")),
        ("283", (6.657, 52.069, 29.10, "HUPSEL")),
        #     ("285", (6.399, 53.575, 0.00, "HUIBERTGAT")),
        ("286", (7.150, 53.196, -0.20, "NIEUW BEERTA")),
        ("290", (6.891, 52.274, 34.80, "TWENTHE")),
        #     ("308", (3.379, 51.381, 0.00, "CADZAND")),
        ("310", (3.596, 51.442, 8.00, "VLISSINGEN")),
        #     ("311", (3.672, 51.379, 0.00, "HOOFDPLAAT")),
        #     ("312", (3.622, 51.768, 0.00, "OOSTERSCHELDE")),
        #     ("313", (3.242, 51.505, 0.00, "VLAKTE V.D. RAAN")),
        #     ("315", (3.998, 51.447, 0.00, "HANSWEERT")),
        #     ("316", (3.694, 51.657, 0.00, "SCHAAR")),
        ("319", (3.861, 51.226, 1.70, "WESTDORPE")),
        ("323", (3.884, 51.527, 1.40, "WILHELMINADORP")),
        #     ("324", (4.006, 51.596, 0.00, "STAVENISSE")),
        ("330", (4.122, 51.992, 11.90, "HOEK VAN HOLLAND")),
        #     ("331", (4.193, 51.480, 0.00, "THOLEN")),
        #     ("340", (4.342, 51.449, 19.20, "WOENSDRECHT")),
        #     ("343", (4.313, 51.893, 3.50, "R'DAM-GEULHAVEN")),
        ("344", (4.447, 51.962, -4.30, "ROTTERDAM")),
        ("348", (4.926, 51.970, -0.70, "CABAUW")),
        ("350", (4.936, 51.566, 14.90, "GILZE-RIJEN")),
        ("356", (5.146, 51.859, 0.70, "HERWIJNEN")),
        ("370", (5.377, 51.451, 22.60, "EINDHOVEN")),
        ("375", (5.707, 51.659, 22.00, "VOLKEL")),
        ("377", (5.763, 51.198, 30.00, "ELL")),
        ("380", (5.762, 50.906, 114.30, "MAASTRICHT")),
        ("391", (6.197, 51.498, 19.50, "ARCEN"))
    ]

    def __init__(self, data_directory='var/data/knmi_measurementstations'):
        self.geoms = utils.multipoint(
            (v[0], v[1]) for _, v in self.STATION_META)
        utils.transform(self.geoms)
        self.data = dict(self._raw_data())
        self.directory = data_directory

    def _raw_data(self):
        for filename in os.listdir(self.directory):
            if 'etmgeg' in filename:
                id_ = filename.replace('etmgeg_', '').replace('.txt', '')
                path = os.path.join(self.directory, filename)
                with open(path, 'r') as f:
                    data = f.read().split(self.HEADER)[1]
                    data_lines = data.split('\n')
                    yield (id_, [
                        [int_or_none(l) for l in line.split(',')]
                        for line in data_lines
                        if line
                    ])

    def closest(self, x, y):
        point = utils.point(x, y)
        i = utils.closest_point(point, self.geoms)
        return self.STATION_META[i]

    def data_from_metadata(self, metadata):
        station_code, meta = metadata
        return self.data[station_code]

    def measurementstation_data(self, x, y):
        return self.data_from_metadata(self.closest(x, y))


# # Code to find out rain / evap datasets:
# RAIN_HEADERS = ['RH', 'RHX', 'RHXH']
# EVAP_HEADERS = ['EV24']
#
# header = [x.strip(' ') for x in HEADER.strip("\n").split(",")]
# dataset_contents = [
#     [dataset[0][0]] +
#     [label for i, label in enumerate(header) if not all(line[i] is None
#                                                     for line in dataset)]
#     for dataset in COLLECTED_DATA
# ]
# evap = [
#     l[0] for l in dataset_contents if any(xx in l for xx in EVAP_HEADERS)]
# rain = [
#     l[0] for l in dataset_contents if any(xx in l for xx in RAIN_HEADERS)]
# # make sure both are equal:
# all(xx == evap[i] for i, xx in rain)

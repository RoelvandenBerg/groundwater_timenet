# Groundwater Timenet
Groundwater timenet is an attempt at deeplearning with (Dutch) groundwater
levels.

## TODOs
##### collect data:
- [X] DINO groundwater levels
- [X] KNMI rain measurement station data
- [X] KNMI rain radar data
- [X] KNMI evapotranspiration
- [X] soil data (Soil physical unit map)
- [X] geohydrology data (GEOTOP data)
- [X] other relevant data
##### parse data:
- [X] geohydrology data (GEOTOP data)
- [X] DINO groundwater levels
- [X] KNMI rain measurement station data
- [X] KNMI evaporation measurement station data
- [X] KNMI rain
- [X] KNMI evapotranspiration
- [ ] soil data
- [ ] other relevant data
##### data exploration:
- [X] geohydrology data (GEOTOP data)
- [ ] DINO groundwater levels
- [X] KNMI rain (use grids)
- [X] KNMI evapotranspiration (use grids)
- [ ] soil data
- [ ] other relevant data
##### other improvements
- [X] cache data in hdf5 files for faster performance
- [ ] normalize data based on data exploration
- [ ] make sure temporal dimensions match
- [ ] create one cached file with all normalized data
##### build a neural network with Keras / Tensorflow:
- [ ] Wavenet atrous CNN architecture
- [ ] LTSM architecture
- [ ] combine / try out different combinations
- [ ] reiterate...

## Data Sources
### Projection
Dutch spatial data uses _Amersfoort / RD New_ projection (ESPG ). Since the datasources
we use use this projection we do not provide an interface that transforms this
data.

### DINO
#### Groundwater stations
We use the BRO groundwater WFS to obtain our groundwaterstation features:
_'http://www.broinspireservices.nl/wfs/osgegmw-a-v1.0'_. This WFS has one
layer: _'Grondwateronderzoek'_. Features in this layer have the following
fields:

- gml_id
- identifier
- dino_nr
- x_rd_crd
- y_rd_crd
- piezometer_nr
- start_date
- end_date
- sample_cnt
- Grondwaterstand|piezometer_nr
- top_depth_mv
- bottom_depth_mv
- top_height_nap
- bottom_height_nap
- Grondwaterstand|start_date
- Grondwaterstand|end_date
- head_cnt
- cluster_id
- cluster_lst
- Grondwatersamenstelling|top_depth_mv
- Grondwatersamenstelling|bottom_depth_mv
- Grondwatersamenstelling|top_height_nap
- Grondwatersamenstelling|bottom_height_nap

Since we are interested in ground water levels, we use the fields:

- dino_nr
- x_rd_crd
- y_rd_crd
- top_depth_mv
- bottom_depth_mv
- top_height_nap
- bottom_height_nap
- Grondwaterstand|start_date
- Grondwaterstand|end_date

An example request for this WFS is:
http://www.broinspireservices.nl/wfs/osgegmw-a-v1.0?SERVICE=WMS&REQUEST=GetFeature&VERSION=2.0.0&TYPENAME=Grondwateronderzoek&SRS=EPSG%3A28992&BBOX=155000.0001213038,352899.52037477936,210050.24017302057,407949.7604122343


#### Groundwater measurements
For groundwater measurements we use the SOAP client that TNO has setup for
Delft-FEWS. Its wsdl file can be obtained from:
_'http://www.dinoservices.nl/gwservices/gws-v11?wsdl'_

This client requires you to specify for each request:

    WELL_NITG_NR='B58D2358',
    START_DATE='1900-01-01',
    END_DATE='2017-12-01',
    UNIT='SFL'

and optionally the `WELL_TUBE_NR`.

#### GeoTOP
We use GeoTOP data for hydrogeology, which can be found [here](
http://www.dinodata.nl/opendap/GeoTOP/contents.html), with [this](
http://www.dinodata.nl/opendap/GeoTOP/codering_geotop.pdf) description. It
is saved as netcdf.

GeoTOP contains data on a 100m x 100m x 0.5m resolution for the whole of the
Netherlands between -50.0m 106.5m NAP.

- x-coordinate in Cartesian system
- y-coordinate in Cartesian system
- diepte t.o.v. NAP
- lithostratigrafie
- meest waarschijnlijke lithoklasse
- kans op het voorkomen van organische stof
- kans op het voorkomen van klei
- kans op het voorkomen van kleiig zand en zandige klei
- kans op het voorkomen van lithoklasse 4
- kans op het voorkomen van zand fijne categorie
- kans op het voorkomen van zand midden categorie
- kans op het voorkomen van zand grove categorie
- kans op het voorkomen van grind
- kans op voorkomen schelpen
- modelonzekerheid lithoklasse
- modelonzekerheid lithostrat


### KNMI
#### Rain
Rain can be downloaded from [this KNMI site](https://data.knmi.nl/datasets/radar_corr_accum_24h/1.0).
And evapotranspiration from [this KNMI site](https://data.knmi.nl/datasets/EV24/2).
Unpack rain and evapotranspiration (years only) respectively in `var/data/rain` and `var/data/et`. This means your folder structure would look something like this:

Because we are interested in timeseries, not in spatial grids, we reshape the data to 3D 10 km matrices with a time dimension to improve performance.

For now these files seem to be broken:
- var/data/rain/2016/03/07RAD_NL25_RAC_24H_201603090800.h5
- var/data/et/1974/12/05/INTER_OPER_R___EV24____L3__19741203T000000_19741204T000000_0002.nc
- var/data/et/1976/10/04/INTER_OPER_R___EV24____L3__19761002T000000_19761003T000000_0002.nc
- var/data/et/1979/01/09/INTER_OPER_R___EV24____L3__19790107T000000_19790108T000000_0002.nc


#### Evapotranspiration
#### Weather station data
Weather station data are downloaded from the [KNMI website](https://www.knmi.nl/nederland-nu/klimatologie/daggegevens). We should be careful using the data however:

> Note: The temperature ranges of the stations The Kooy, Eelde, De Bilt, Vlissingen and Beek are homogenized. The series of other stations are not homogenized. Due to station relocations and changes in observation methods, those time series of day values may be inhomogeneous. This means that these series are not suitable for trend analysis. For studies on climate change, we refer to the homogenized series or the Central Netherlands Temperature. Trend analyzes are presented in the KNMI-14 climate scenario brochure.

Metadata for the stations can be found [here](http://projects.knmi.nl/klimatologie/metadata/index.html).

Description from file:

    SOURCE: ROYAL NETHERLANDS METEOROLOGICAL INSTITUTE (KNMI)
    Comment: These time series are inhomogeneous because of station relocations and changes in observation techniques. As a result, these series are not suitable for trend analysis. For climate change studies we refer to the homogenized series of monthly temperatures of De Bilt <http://www.knmi.nl/kennis-en-datacentrum/achtergrond/gehomogeniseerde-reeks-maandtemperaturen-de-bilt> or the Central Netherlands Temperature <http://www.knmi.nl/kennis-en-datacentrum/achtergrond/centraal-nederland-temperatuur-cnt>.

    YYYYMMDD  = Datum (YYYY=jaar MM=maand DD=dag) / Date (YYYY=year MM=month DD=day)
    DDVEC     = Vectorgemiddelde windrichting in graden (360=noord, 90=oost, 180=zuid, 270=west, 0=windstil/variabel). Zie http://www.knmi.nl/kennis-en-datacentrum/achtergrond/klimatologische-brochures-en-boeken / Vector mean wind direction in degrees (360=north, 90=east, 180=south, 270=west, 0=calm/variable)
    FHVEC     = Vectorgemiddelde windsnelheid (in 0.1 m/s). Zie http://www.knmi.nl/kennis-en-datacentrum/achtergrond/klimatologische-brochures-en-boeken / Vector mean windspeed (in 0.1 m/s)
    FG        = Etmaalgemiddelde windsnelheid (in 0.1 m/s) / Daily mean windspeed (in 0.1 m/s)
    FHX       = Hoogste uurgemiddelde windsnelheid (in 0.1 m/s) / Maximum hourly mean windspeed (in 0.1 m/s)
    FHXH      = Uurvak waarin FHX is gemeten / Hourly division in which FHX was measured
    FHN       = Laagste uurgemiddelde windsnelheid (in 0.1 m/s) / Minimum hourly mean windspeed (in 0.1 m/s)
    FHNH      = Uurvak waarin FHN is gemeten / Hourly division in which FHN was measured
    FXX       = Hoogste windstoot (in 0.1 m/s) / Maximum wind gust (in 0.1 m/s)
    FXXH      = Uurvak waarin FXX is gemeten / Hourly division in which FXX was measured
    TG        = Etmaalgemiddelde temperatuur (in 0.1 graden Celsius) / Daily mean temperature in (0.1 degrees Celsius)
    TN        = Minimum temperatuur (in 0.1 graden Celsius) / Minimum temperature (in 0.1 degrees Celsius)
    TNH       = Uurvak waarin TN is gemeten / Hourly division in which TN was measured
    TX        = Maximum temperatuur (in 0.1 graden Celsius) / Maximum temperature (in 0.1 degrees Celsius)
    TXH       = Uurvak waarin TX is gemeten / Hourly division in which TX was measured
    T10N      = Minimum temperatuur op 10 cm hoogte (in 0.1 graden Celsius) / Minimum temperature at 10 cm above surface (in 0.1 degrees Celsius)
    T10NH     = 6-uurs tijdvak waarin T10N is gemeten / 6-hourly division in which T10N was measured; 6=0-6 UT, 12=6-12 UT, 18=12-18 UT, 24=18-24 UT
    SQ        = Zonneschijnduur (in 0.1 uur) berekend uit de globale straling (-1 voor <0.05 uur) / Sunshine duration (in 0.1 hour) calculated from global radiation (-1 for <0.05 hour)
    SP        = Percentage van de langst mogelijke zonneschijnduur / Percentage of maximum potential sunshine duration
    Q         = Globale straling (in J/cm2) / Global radiation (in J/cm2)
    DR        = Duur van de neerslag (in 0.1 uur) / Precipitation duration (in 0.1 hour)
    RH        = Etmaalsom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm) / Daily precipitation amount (in 0.1 mm) (-1 for <0.05 mm)
    RHX       = Hoogste uursom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm) / Maximum hourly precipitation amount (in 0.1 mm) (-1 for <0.05 mm)
    RHXH      = Uurvak waarin RHX is gemeten / Hourly division in which RHX was measured
    PG        = Etmaalgemiddelde luchtdruk herleid tot zeeniveau (in 0.1 hPa) berekend uit 24 uurwaarden / Daily mean sea level pressure (in 0.1 hPa) calculated from 24 hourly values
    PX        = Hoogste uurwaarde van de luchtdruk herleid tot zeeniveau (in 0.1 hPa) / Maximum hourly sea level pressure (in 0.1 hPa)
    PXH       = Uurvak waarin PX is gemeten / Hourly division in which PX was measured
    PN        = Laagste uurwaarde van de luchtdruk herleid tot zeeniveau (in 0.1 hPa) / Minimum hourly sea level pressure (in 0.1 hPa)
    PNH       = Uurvak waarin PN is gemeten / Hourly division in which PN was measured
    VVN       = Minimum opgetreden zicht / Minimum visibility; 0: <100 m, 1:100-200 m, 2:200-300 m,..., 49:4900-5000 m, 50:5-6 km, 56:6-7 km, 57:7-8 km,..., 79:29-30 km, 80:30-35 km, 81:35-40 km,..., 89: >70 km)
    VVNH      = Uurvak waarin VVN is gemeten / Hourly division in which VVN was measured
    VVX       = Maximum opgetreden zicht / Maximum visibility; 0: <100 m, 1:100-200 m, 2:200-300 m,..., 49:4900-5000 m, 50:5-6 km, 56:6-7 km, 57:7-8 km,..., 79:29-30 km, 80:30-35 km, 81:35-40 km,..., 89: >70 km)
    VVXH      = Uurvak waarin VVX is gemeten / Hourly division in which VVX was measured
    NG        = Etmaalgemiddelde bewolking (bedekkingsgraad van de bovenlucht in achtsten, 9=bovenlucht onzichtbaar) / Mean daily cloud cover (in octants, 9=sky invisible)
    UG        = Etmaalgemiddelde relatieve vochtigheid (in procenten) / Daily mean relative atmospheric humidity (in percents)
    UX        = Maximale relatieve vochtigheid (in procenten) / Maximum relative atmospheric humidity (in percents)
    UXH       = Uurvak waarin UX is gemeten / Hourly division in which UX was measured
    UN        = Minimale relatieve vochtigheid (in procenten) / Minimum relative atmospheric humidity (in percents)
    UNH       = Uurvak waarin UN is gemeten / Hourly division in which UN was measured
    EV24      = Referentiegewasverdamping (Makkink) (in 0.1 mm) / Potential evapotranspiration (Makkink) (in 0.1 mm)

### Soil (WUR / Alterra)
The [Dutch 1:50.000 soil map](http://www.wur.nl/nl/show/Bodemkaart-1-50-000.htm) does not seem to be free.
The Soil physical unit map ([Bodemfysische-Eenhedenkaart](http://www.wur.nl/nl/show/Bodemfysische-Eenhedenkaart-BOFEK2012.htm)) is however. Which suits us just as well for now.

### Other
#### Deltares:
- [Irrigation](https://data.overheid.nl/data/dataset/irrigatiewater-locatie-beregeningsonttrekkingen-uit-grondwater-en-oppervlaktewater)
- [Drinking water](https://data.overheid.nl/data/dataset/drinkwater)

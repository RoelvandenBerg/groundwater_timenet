# Groundwater Timenet
Groundwater timenet is an attempt at deeplearning with (Dutch) groundwater 
levels. 

## TODOs
##### collect data:
- [X] DINO groundwater levels
- [ ] KNMI rain
- [ ] rain radar data? 
- [ ] KNMI evaporation
- [ ] soil data
- [X] geohydrology data (GEOTOP data)
##### parse data:
- [X] DINO groundwater levels
- [ ] KNMI rain
- [ ] rain radar data? 
- [ ] KNMI evaporation
- [ ] soil data
- [X] geohydrology data (GEOTOP data)
##### build a neural network with Keras / Tensorflow:
- [ ] Wavenet atrous CNN architecture
- [ ] LTSM architecture
- [ ] combine / try out different combinations
- [ ] reiterate...


## Data Sources
### Projection
Dutch spatial data uses _Amersfoort / RD New_ projection. Since the datasources 
we use use this projection we do not provide an interface that transforms this 
data. If we would want to use this. 

### DINO
#### Groundwater stations
We use the BRO groundwater WFS to obtain our groundwaterstation features:  
_'http://www.broinspireservices.nl/wfs/osgegmw-a-v1.0'_. This WFS has one 
layer: _'Grondwateronderzoek'_. Features in this layer have the following 
fields:
    
    ['gml_id',
     'identifier',
     'dino_nr',
     'x_rd_crd',
     'y_rd_crd',
     'piezometer_nr',
     'start_date',
     'end_date',
     'sample_cnt',
     'Grondwaterstand|piezometer_nr',
     'top_depth_mv',
     'bottom_depth_mv',
     'top_height_nap',
     'bottom_height_nap',
     'Grondwaterstand|start_date',
     'Grondwaterstand|end_date',
     'head_cnt',
     'cluster_id',
     'cluster_lst',
     'Grondwatersamenstelling|top_depth_mv',
     'Grondwatersamenstelling|bottom_depth_mv',
     'Grondwatersamenstelling|top_height_nap',
     'Grondwatersamenstelling|bottom_height_nap']

Since we are interested in ground water levels, we use the fields:

    ['dino_nr',
     'x_rd_crd',
     'y_rd_crd',
     'top_depth_mv',
     'bottom_depth_mv',
     'top_height_nap',
     'bottom_height_nap',
     'Grondwaterstand|start_date',
     'Grondwaterstand|end_date']

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
#### Evaporation

### Soil
Use Dutch 1:50.000 soil map.


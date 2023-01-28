# thames-sewage
Realtime mapping of sewage release events in Thames basin

## Installation 

This script uses depends on another package [`autocatchments`](https://github.com/AlexLipp/autocatchments), and other standard scientific and geospatial packages. 

## Usage 

Run `python make_discharge_map.py` will generate a time-stamped map of the number of upstream sewage discharge events at any point in the Thames basin. It outputs the results as a `geoJSON`.

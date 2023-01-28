# thames-sewage
Realtime mapping of sewage release events in Thames basin

## Installation 

In addition to the dependencies listed in `requirements.txt`, this script depends on the custom-made package [`autocatchments`](https://github.com/AlexLipp/autocatchments) which was created initially for personal use only. 

## Usage 

Running `python make_discharge_map.py` from the command line will generate a time-stamped map of the number of upstream sewage discharge events at any point in the Thames basin. It additionall outputs the results as a `geoJSON` which can be loaded in to any standard GIS environment.

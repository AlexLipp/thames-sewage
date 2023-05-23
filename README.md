
# Sewage Discharges in the Thames Basin

Realtime mapping the downstream impact of Combined Sewage Overflow discharge events in the Thames basin. This repository provides the back-end for [`www.sewagemap.co.uk`](https://www.sewagemap.co.uk/). 

This was developed by [Alex Lipp](https://github.com/AlexLipp), [Sudhir Balaji](https://github.com/sudhir-b) and others. Please feel free to raise an issue above or contact us directly 

![Twitter Follow](https://img.shields.io/twitter/follow/AlexGLipp?style=social)

## Installation 

In addition to the dependencies listed in `requirements.txt`, this script depends on the custom-made package [`autocatchments`](https://github.com/AlexLipp/autocatchments). The file `environment.yaml` is a conda environment file that can be used to create a conda environment with the necessary requirements. 

## Usage 

Running `python make_discharge_map.py` from the command line will generate a time-stamped map of the number of upstream sewage discharge events at any point in the Thames basin. An example of this output is shown below. It additionally outputs the results as a `geoJSON` which can be loaded in to any standard GIS environment. These `geoJSON` files are used to display the real-time sewage-map at [`www.sewagemap.co.uk`](https://www.sewagemap.co.uk/).   Note that running the scripts requires having local versions of the input Digital Elevation Model which is too large to store on GitHub. Please contact [me](https://github.com/AlexLipp) if you'd like to access these. ![28123_2032](https://user-images.githubusercontent.com/10188895/215289603-3315e7b6-5a50-48ed-9ef0-7a9269e5e2e3.png).

## Contact 


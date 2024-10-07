
# Sewage Discharges in the Thames Basin

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


Realtime mapping the downstream impact of Combined Sewage Overflow discharge events in the Thames basin. This repository provides the back-end for [`www.sewagemap.co.uk`](https://www.sewagemap.co.uk/). The repository  for the front-end is available at [`github.com/JonnyDawe/UK-Sewage-Map/`](https://github.com/JonnyDawe/UK-Sewage-Map/).

This was developed by [Alex Lipp](https://www.merton.ox.ac.uk/people/dr-alexander-lipp), [Jonny Dawe](https://www.linkedin.com/in/jonathan-dawe-46180212a) and Sudhir Balaji. Please feel free to raise an issue above or contact us directly.

[![Twitter Follow](https://img.shields.io/twitter/follow/alexglipp?style=social)](https://twitter.com/intent/follow?screen_name=AlexGLipp)
[![Twitter Follow](https://img.shields.io/twitter/follow/JdMapDev?style=social)](https://twitter.com/intent/follow?screen_name=JdMapDev)
[![GitHub followers](https://img.shields.io/github/followers/AlexLipp?label=AlexLipp&style=social)](https://github.com/AlexLipp)
[![GitHub followers](https://img.shields.io/github/followers/JonnyDawe?label=JonnyDawe&style=social)](https://github.com/JonnyDawe)
[![GitHub followers](https://img.shields.io/github/followers/sudhir-b?label=sudhir-b&style=social)](https://github.com/sudhir-b)

## Installation

This script relies on the `POOPy` package which I have created to allow easy interaction with Water Company Event Duration Monitoring APIs, and analysis of the data. This is also freely available at: [`github.com/AlexLipp/POOPy`](https://github.com/AlexLipp/POOPy).

To access the data stored in the Thames Water API you will need to register for the API [here](https://www.thameswater.co.uk/about-us/performance/river-health/storm-discharge-data#third-party-api). The script expects the API keys to be stored as environment variables (details given in the [POOPy](https://github.com/AlexLipp/POOPy) repository).

## Usage

The core script is `update.py` which is called automatically every 15 minutes. This function, using POOPy functions, calculates a geoJSON file which contains the downstream impact of all active or recently active CSO spills in the Thames Basin. Additionally, it creates a JSON file which contains the history of all discharges for all monitors. These are automatically uploaded to the Amazon Web Services bucket which hosts them. The data is then fronted using the CloudFront delivery service. The files are read by the `www.sewagemap.co.uk` front-end which visualises them.

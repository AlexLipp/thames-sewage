
# Sewage Discharges in the Thames Basin

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


Realtime mapping the downstream impact of Combined Sewage Overflow discharge events in the Thames basin. This repository provides the back-end for [`www.sewagemap.co.uk`](https://www.sewagemap.co.uk/).

This was developed by [Alex Lipp](https://www.merton.ox.ac.uk/people/dr-alexander-lipp), [Jonny Dawe](https://www.linkedin.com/in/jonathan-dawe-46180212a) and Sudhir Balaji. Please feel free to raise an issue above or contact us directly.

[![Twitter Follow](https://img.shields.io/twitter/follow/alexglipp?style=social)](https://twitter.com/intent/follow?screen_name=AlexGLipp)
[![Twitter Follow](https://img.shields.io/twitter/follow/JdMapDev?style=social)](https://twitter.com/intent/follow?screen_name=JdMapDev)
[![GitHub followers](https://img.shields.io/github/followers/AlexLipp?label=AlexLipp&style=social)](https://github.com/AlexLipp)
[![GitHub followers](https://img.shields.io/github/followers/JonnyDawe?label=JonnyDawe&style=social)](https://github.com/JonnyDawe)
[![GitHub followers](https://img.shields.io/github/followers/sudhir-b?label=sudhir-b&style=social)](https://github.com/sudhir-b)

## Installation

This script uses the `POOPy` package which allows easily interaction with Water Company EDM APIs, and analysis of the data. This is available at: [`github.com/AlexLipp/POOPy`](https://github.com/AlexLipp/POOPy).

To access the data stored in the Thames Water API you will need to register for the API [here](https://www.thameswater.co.uk/about-us/performance/river-health/storm-discharge-data#third-party-api). The script expects the API keys to be stored as environment variables (details given in the script).

## Usage

The core script is `update.py` which is called automatically every 15 minutes. This function calculates a geoJSON file which contains the downstream impact of all active or recently active CSO in the Thames Basin. Additionally, it creates a JSON file which contains the history of all discharges for all monitors. These are automatically uploaded to the AWS bucket which hosts them. These are then read by the `www.sewagemap.co.uk` front-end which visualises them.

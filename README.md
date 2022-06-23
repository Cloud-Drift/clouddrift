# CloudDrift
![CI](https://github.com/Cloud-Drift/clouddrift/workflows/CI/badge.svg)
![Documentation Status](https://github.com/Cloud-Drift/clouddrift/actions/workflows/docs.yml/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cloud-Drift/clouddrift/main?labpath=examples)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FCloud-Drift%2Fclouddrift&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

This project aims at accelerating the use of Lagrangian data for atmospheric, oceanic, and climate sciences. It is currently funded by the [NSF EarthCube program](https://www.earthcube.org/info). The lead Principal Investigator of the project is [Shane Elipot](https://github.com/selipot) and its developement is led by [Philippe Miron](https://github.com/philippemiron).

This project is based on the open source software python. This software requires a number of modules that need to exist within your working python environment. The list of necessary modules is contained in the YAML file `environment.yml`. This file can be used with a package manager such as conda to create the necessary python environment. From the command line, run

`conda env create --file environment.yml`

Once the library is in a *beta* stage, a conda/pip packages will be available to simplify the installation process.

This repository is organized as follows:
- `cloudrift/`: modules of the clouddrift library
- `data/`: processing scripts for various Lagrangian datasets, including the GDP historical dataset.
- `docs/`: documentation *soon* available at [clouddrift.readthedocs.io](clouddrift.readthedocs.io)
- `examples/`: series of Jupyter Notebooks showcasing the library use cases
- `tests/`: test-suite
- `dev/`: developement Jupyter Notebooks use to brainstorm new ideas and discussions

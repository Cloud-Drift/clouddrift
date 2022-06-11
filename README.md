# CloudDrift

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

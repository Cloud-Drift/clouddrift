# CloudDrift
![CI](https://github.com/Cloud-Drift/clouddrift/workflows/CI/badge.svg)
[![Documentation Status](https://github.com/Cloud-Drift/clouddrift/actions/workflows/docs.yml/badge.svg)](https://cloud-drift.github.io/clouddrift)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cloud-Drift/clouddrift-examples/main?labpath=notebooks)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NSF-2126413](https://img.shields.io/badge/NSF-2126413-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2126413)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FCloud-Drift%2Fclouddrift&count_bg=%2368C563&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

CloudDrift accelerates the use of Lagrangian data for atmospheric, oceanic, and climate sciences.
It is funded by [NSF EarthCube](https://www.earthcube.org/info) through the
[EarthCube Capabilities Grant No. 2126413](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2126413).

## Getting started

### Install CloudDrift

You can install the latest release of CloudDrift using pip or Conda.
You can also install the latest development (unreleased) version from GitHub.

#### pip

In your virtual environment, type:

```
pip install clouddrift
```

#### Conda

First add `conda-forge` to your channels in your Conda environment:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

then install CloudDrift:

```
conda install clouddrift
```

#### Development branch

If you need the latest development version, get it from GitHub using pip:

```
pip install git+https://github.com/cloud-drift/clouddrift
```

### Run the tests

To run the tests, you need to first download the CloudDrift source code from
GitHub and install it in your virtual environment:

```
git clone https://github.com/cloud-drift/clouddrift
cd clouddrift
python3 -m venv venv
source venv/bin/activate
pip install .
```

Then, run the tests like this:

```
python -m unittest tests/*.py
```

## Using CloudDrift

Start by reading the [documentation](https://cloud-drift.github.io/clouddrift).

Example Jupyter notebooks that showcase the library, as well as scripts
to process various Lagrangian datasets, are in
[clouddrift-examples](https://github.com/Cloud-Drift/clouddrift-examples).

## Found an issue or need help?

Please create a new issue [here](https://github.com/Cloud-Drift/clouddrift/issues/new)
and provide as much detail as possible about your problem or question.

# CloudDrift
![CI](https://github.com/Cloud-Drift/clouddrift/workflows/CI/badge.svg)
[![Documentation Status](https://github.com/Cloud-Drift/clouddrift/actions/workflows/docs.yml/badge.svg)](https://cloud-drift.github.io/clouddrift)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cloud-Drift/clouddrift-examples/main?labpath=notebooks)
[![Available on conda-forge](https://img.shields.io/badge/Anaconda.org-0.6.0-blue.svg)](https://anaconda.org/conda-forge/clouddrift/)
[![Available on pypi](https://img.shields.io/pypi/v/clouddrift.svg?style=flat&color=blue)](https://pypi.org/project/clouddrift/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NSF-2126413](https://img.shields.io/badge/NSF-2126413-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2126413)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FCloud-Drift%2Fclouddrift&count_bg=%2368C563&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

CloudDrift accelerates the use of Lagrangian data for atmospheric, oceanic, and climate sciences.
It is funded by [NSF EarthCube](https://www.earthcube.org/info) through the
[EarthCube Capabilities Grant No. 2126413](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2126413).

## Getting started

### Install CloudDrift

You can install the latest release of CloudDrift using pip or conda. Note that using the pip [package](https://pypi.org/project/clouddrift/), you might have to manually install system libraries required by `clouddrift`, e.g. the `libnetcdf4`. As an alternative, using the conda [package](https://anaconda.org/conda-forge/clouddrift) those libraries are automatically installed.

#### Latest official release
##### pip

In your virtual environment, type:

```
pip install clouddrift
```

##### Conda

First add `conda-forge` to your channels in your Conda configuration (`~/.condarc`):

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

then install CloudDrift:

```
conda install clouddrift
```

#### Development branch

If you need the latest development version, you can install it directly from GitHub.

##### pip

```
pip install git+https://github.com/cloud-drift/clouddrift
```

##### Conda
```
conda env create -f environment.yml
````
with the environment [file](https://github.com/Cloud-Drift/clouddrift/blob/main/environment.yml) located in the main repository.

### Run the tests

To run the tests, you need to first download the CloudDrift source code from
GitHub and create the required environment:

```
git clone https://github.com/cloud-drift/clouddrift
cd clouddrift/
conda env create -f environment.yml
conda activate clouddrift
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

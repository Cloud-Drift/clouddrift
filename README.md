# CloudDrift
![CI](https://github.com/Cloud-Drift/clouddrift/workflows/CI/badge.svg)
[![Documentation Status](https://github.com/Cloud-Drift/clouddrift/actions/workflows/docs.yml/badge.svg)](https://cloud-drift.github.io/clouddrift)
[![codecov](https://codecov.io/gh/Cloud-Drift/clouddrift/branch/main/graph/badge.svg)](https://codecov.io/gh/Cloud-Drift/clouddrift/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cloud-Drift/clouddrift-examples/main?labpath=notebooks)
[![Available on conda-forge](https://anaconda.org/conda-forge/clouddrift/badges/version.svg?style=flat-square)](https://anaconda.org/conda-forge/clouddrift/)
[![Available on pypi](https://img.shields.io/pypi/v/clouddrift.svg?style=flat-square&color=blue)](https://pypi.org/project/clouddrift/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![NSF-2126413](https://img.shields.io/badge/NSF-2126413-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2126413)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11081648.svg)](https://doi.org/10.5281/zenodo.11081648)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FCloud-Drift%2Fclouddrift&count_bg=%2368C563&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

CloudDrift is a Python package that accelerates the use of Lagrangian data for atmospheric, oceanic, and climate sciences.
It is funded by [NSF EarthCube](https://www.earthcube.org/info) through the
[EarthCube Capabilities Grant No. 2126413](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2126413).

Read the [documentation](https://cloud-drift.github.io/clouddrift) or explore
the [Jupyter Notebook Examples](https://github.com/Cloud-Drift/clouddrift-examples).

## Using CloudDrift

Start by reading the [documentation](https://cloud-drift.github.io/clouddrift).

Example Jupyter notebooks that showcase the library, as well as scripts
to process various Lagrangian datasets, can be found in
[clouddrift-examples](https://github.com/Cloud-Drift/clouddrift-examples), [gdp-get-started](https://github.com/Cloud-Drift/gdp-get-started), [mosaic-get-started](https://github.com/Cloud-Drift/mosaic-get-started), or [a demo for the EarthCube community workshop 2023](https://github.com/Cloud-Drift/e3-comm-workshop-2023).

## Contributing and scope

We welcome and invite contributions from the community in any shape or form! Please visit our [Contributing Guide](CONTRIBUTING.md) to get Started 😃

The scope of CloudDrift includes:

* Working with contiguous ragged-array data; for example, see the
  [`clouddrift.ragged`](https://cloud-drift.github.io/clouddrift/_autosummary/clouddrift.ragged.html) module.
* Common scientific analysis of Lagrangian data, oceanographic or otherwise;
  for example, see the
  [`clouddrift.kinematics`](https://cloud-drift.github.io/clouddrift/_autosummary/clouddrift.kinematics.html),
  [`clouddrift.signal`](https://cloud-drift.github.io/clouddrift/_autosummary/clouddrift.signal.html), and
  [`clouddrift.wavelet`](https://cloud-drift.github.io/clouddrift/_autosummary/clouddrift.wavelet.html) modules.
* Processing existing Lagrangian datasets into a common data structure and format;
  for example, see the [`clouddrift.adapters.mosaic`](https://cloud-drift.github.io/clouddrift/_autosummary/clouddrift.adapters.mosaic.html) module.
* Making cloud-optimized ragged-array datasets easily accessible; for example,
  see the [`clouddrift.datasets`](https://cloud-drift.github.io/clouddrift/_autosummary/clouddrift.datasets.html) module.

If you have an idea that does not fit into the scope of CloudDrift but you think
it should, please open an issue to discuss it.

## Getting started

### Install CloudDrift

You can install the latest release of CloudDrift using [pip](https://pypi.org/project/clouddrift/) or [conda](https://anaconda.org/conda-forge/clouddrift).

#### Latest official release:
##### pip:

In your virtual environment, type:

```
pip install clouddrift
```

To install optional dependencies needed by the `clouddrift.plotting` module,
type:

```
pip install matplotlib cartopy
```

##### Conda:

First add `conda-forge` to your channels in your Conda configuration (`~/.condarc`):

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

then install CloudDrift:

```
conda install clouddrift
```

To install optional dependencies needed by the `clouddrift.plotting` module,
type:

```
conda install matplotlib-base cartopy
```

#### Development branch:

If you need the latest development version, you can install it directly from this GitHub repository.

##### pip:

In your virtual environment, type:

```
pip install git+https://github.com/cloud-drift/clouddrift
```

##### Conda:
```
conda env create -f environment.yml
```
with the environment [file](https://github.com/Cloud-Drift/clouddrift/blob/main/environment.yml) located in the main repository.

### Installing CloudDrift on unsupported platforms

One or more dependencies of CloudDrift may not have pre-built wheels for
platforms like IBM Power9 or Raspberry Pi.
If you are using pip to install CloudDrift and are getting errors during the
installation step, try installing CloudDrift using Conda.
If you still have issues installing CloudDrift, you may need to install system
dependencies first.
Please let us know by opening an
[issue](https://github.com/Cloud-Drift/clouddrift/issues/new) and we will do our
best to help you.

## Found an issue or need help?

Please create a new issue [here](https://github.com/Cloud-Drift/clouddrift/issues/new)
and provide as much detail as possible about your problem or question.

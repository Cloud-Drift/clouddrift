# `clouddrift`
![CI](https://github.com/Cloud-Drift/clouddrift/workflows/CI/badge.svg)
[![Documentation Status](https://github.com/Cloud-Drift/clouddrift/actions/workflows/docs.yml/badge.svg)](https://cloud-drift.github.io/clouddrift)
[![codecov](https://codecov.io/gh/Cloud-Drift/clouddrift/branch/main/graph/badge.svg)](https://codecov.io/gh/Cloud-Drift/clouddrift/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![NSF-2126413](https://img.shields.io/badge/NSF-2126413-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=2126413)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11081647.svg)](https://doi.org/10.5281/zenodo.11081647)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FCloud-Drift%2Fclouddrift&count_bg=%2368C563&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

### 📦 Distributions
[![Available on conda-forge](https://anaconda.org/conda-forge/clouddrift/badges/version.svg?style=flat-square)](https://anaconda.org/conda-forge/clouddrift/)
[![Available on pypi](https://img.shields.io/pypi/v/clouddrift.svg?style=flat-square&color=blue)](https://pypi.org/project/clouddrift/)

### 👥 Social
[![Bluesky](https://img.shields.io/badge/Bluesky-0285FF?logo=bluesky&logoColor=fff&style=for-the-badge)](https://bsky.app/profile/clouddrift.bsky.social)
[![Twitter/X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/CloudDrift2)

### 📚 Binders and examples
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cloud-Drift/mosaic-get-started/HEAD) : **MOSAiC get started (🧊 sea ice trajectories)**

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cloud-Drift/gdp-get-started/HEAD) : **GDP get started (🌊 ocean drifter trajectories)**

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Cloud-Drift/hurdat2-get-started/HEAD) : **HURDAT2 get started (🌀 cyclone/hurricane trajectories from 1852 - 2022)**

- [HYCOM-OceanTrack](https://github.com/selipot/hycom-oceantrack): A repository with notebook examples using `clouddrift` with a very large , analysis-ready cloud-optimized, Lagrangian dataset hosted in the cloud: [HYCOM OceanTrack: Integrated HYCOM Eulerian Fields and Lagrangian Trajectories Dataset](https://registry.opendata.aws/hycom-global-drifters/index.html).


`clouddrift` is a Python package that accelerates the use of Lagrangian data for atmospheric, oceanic, and climate sciences.
It is funded by [NSF EarthCube](https://www.earthcube.org/info) through the
[EarthCube Capabilities Grant No. 2126413](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2126413).

Read the [documentation](https://cloud-drift.github.io/clouddrift).

## Using `clouddrift`

Start by reading the [documentation](https://cloud-drift.github.io/clouddrift).

Example Jupyter notebooks that showcase the library, as well as scripts
to process various Lagrangian datasets, can be found in [gdp-get-started](https://github.com/Cloud-Drift/gdp-get-started), [mosaic-get-started](https://github.com/Cloud-Drift/mosaic-get-started), [hurdat2-get-started](https://github.com/Cloud-Drift/hurdat2-get-started), or [a demo for the EarthCube community workshop 2023](https://github.com/Cloud-Drift/e3-comm-workshop-2023).

## Contributing and scope

We welcome and invite contributions from the community in any shape or form! Please visit our [Contributing Guide](CONTRIBUTING.md) to get Started 😃

The scope of `clouddrift` includes:

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

If you have an idea that does not fit into the scope of `clouddrift` but you think
it should, please open an issue to discuss it.

## Getting started

### Install `clouddrift`

You can install the latest release of `clouddrift` using [pip](https://pypi.org/project/clouddrift/) or [conda](https://anaconda.org/conda-forge/clouddrift).

#### Latest official release:
##### pip:

In your virtual environment, type:

```
pip install clouddrift
```

To install optional dependencies needed by the `clouddrift.plotting` module,
type:

```
pip install clouddrift[plotting]
```

##### Conda:

First add `conda-forge` to your channels in your Conda configuration (`~/.condarc`):

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

then install `clouddrift`:

```
conda install clouddrift
```

To install optional dependencies needed by the `clouddrift.plotting` module,
type:

```
conda install matplotlib cartopy
```

#### Development branch:

If you need the latest development version, you can install it directly from this GitHub repository.

##### pip:

In your existing virtual environment, you can use `pip` as follows.
1. Get the code:

```bash
git clone https://github.com/cloud-drift/clouddrift
cd clouddrift/
```

2. Install dependencies and local version of `clouddrift`:
```bash
pip install .
```

##### Conda:

Using `conda`, you can proceed as follows.

1. Get the code:

```bash
git clone https://github.com/cloud-drift/clouddrift
cd clouddrift/
```

2. Create an environment as specified in the yml file with the required library dependencies:
```bash
conda env create -f environment.yml # creates a new env with the dependencies
conda env update -f environment.yml # install dependencies in current environment
```

2a. Make sure you **created** the environment by activating it:
```bash
conda activate clouddrift
```

3. Finally, install the local version of `clouddrift`:
```bash
pip install .
```

### Installing `clouddrift` on unsupported platforms

One or more dependencies of `clouddrift` may not have pre-built wheels for
platforms like IBM Power9 or Raspberry Pi.
If you are using pip to install `clouddrift` and are getting errors during the
installation step, try installing `clouddrift` using Conda.
If you still have issues installing `clouddrift`, you may need to install system
dependencies first.
Please let us know by opening an
[issue](https://github.com/Cloud-Drift/clouddrift/issues/new) and we will do our
best to help you.

## Found an issue or need help?

Please create a new issue [here](https://github.com/Cloud-Drift/clouddrift/issues/new)
and provide as much detail as possible about your problem or question.

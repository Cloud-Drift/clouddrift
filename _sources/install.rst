.. _install:

Installation
============

Required dependencies
---------------------

CloudDrift requires the following dependencies:

- `python <https://github.com/python>`_ (3.10 or later)
- `aiohttp <https://github.com/aio-libs/aiohttp>`_ (3.8.4 or later)
- `awkward <https://github.com/scikit-hep/awkward>`_ (2.0.0 or later)
- `fsspec <https://github.com/fsspec/filesystem_spec>`_ (2022.3.0 or later)
- `netcdf4 <https://github.com/Unidata/netcdf4-python>`_ (1.6.4 or later)
- `h5netcdf <https://github.com/h5netcdf/h5netcdf>`_ (1.3.0 or later)
- `numpy <https://github.com/numpy/numpy>`_ (1.22.4 or later)
- `pandas <https://github.com/pandas-dev/pandas>`_ (1.3.4 or later)
- `pyarrow <https://github.com/apache/arrow>`_ (8.0.0 or later)
- `tqdm <https://github.com/tqdm/tqdm>`_ (4.64.0 or later)
- `requests <https://github.com/psf/requests>`_ (2.31.0 or later)
- `scipy <https://github.com/scipy/scipy>`_ (1.11.2 or later)
- `xarray <https://github.com/pydata/xarray>`_ (2023.5.0 or later)
- `zarr <https://github.com/zarr-developers/zarr-python>`_ (2.14.2 or later)
- `tenacity <https://github.com/jd/tenacity>`_ (8.2.3 or later)

if you install CloudDrift using pip or Conda, these dependencies will be installed automatically.

Optional dependencies
---------------------

For plotting
^^^^^^^^^^^^

- `cartopy <https://github.com/SciTools/cartopy>`_
- `matplotlib <https://github.com/matplotlib/matplotlib>`_

For development and testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `build <https://github.com/pypa/build>`_
- `coverage <https://github.com/nedbat/coveragepy>`_
- `docutils <https://github.com/docutils/docutils>`_
- `pytest <https://github.com/pytest-dev/pytest>`_
- `mypy <https://github.com/python/mypy>`_
- `ruff <https://github.com/astral-sh/ruff>`_
- `twine <https://github.com/pypa/twine>`_

For building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `sphinx <https://github.com/sphinx-doc/sphinx>`_
- `sphinx-book-theme <https://github.com/executablebooks/sphinx-book-theme>`_
- `sphinx-copybutton <https://github.com/executablebooks/sphinx-copybutton>`_

Note: If you are using pip to install xarray, optional dependencies can be installed by specifying extras, such as:

.. code-block:: text

  pip install clouddrift[plotting] clouddrift[dev] clouddrift[docs]

There is also `clouddrift[all]` to install automatically all optional dependencies.

Installation instructions
-------------------------

You can install the latest release of CloudDrift using pip or Conda.
You can also install the latest development (unreleased) version from GitHub.

pip
^^^

In your virtual environment, type:

.. code-block:: text

  pip install clouddrift

To install optional dependencies needed by the ``clouddrift.plotting`` module,
type:

.. code-block:: text

  pip install clouddrift[plotting]

Conda
^^^^^

First add ``conda-forge`` to your channels in your Conda environment:

.. code-block:: text

  conda config --add channels conda-forge
  conda config --set channel_priority strict

then install CloudDrift:

.. code-block:: text

  conda install clouddrift

To install optional dependencies needed by the ``clouddrift.plotting`` module,
type:

.. code-block:: text

  conda install matplotlib cartopy

Developers
^^^^^^^^^^

If you need the latest development version, get it from GitHub using pip:

.. code-block:: text

  pip install git+https://github.com/Cloud-Drift/clouddrift

Running tests
=============

To run the tests, you need to first download the CloudDrift source code from
GitHub and install it in your virtual environment:


.. code-block:: text

  git clone https://github.com/cloud-drift/clouddrift
  cd clouddrift
  python3 -m venv venv
  source venv/bin/activate
  pip install .

Then, run the tests like this:

.. code-block:: text

  python -m unittest tests/*.py

A quick how-to guide is provided on the `Usage <https://cloud-drift.github.io/clouddrift/usage.html>`_ page.

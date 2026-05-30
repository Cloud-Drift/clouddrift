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

if you install CloudDrift using uv, pip or Conda, these dependencies will be installed automatically. See :ref:`installation-instructions` for details.

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

Note: Optional plotting dependencies can be installed by specifying extras:

- with uv: ``uv add 'clouddrift[plotting]'``
- with pip: ``pip install 'clouddrift[plotting]'``

There is also ``clouddrift[all]`` to install all optional dependencies.

.. _installation-instructions:

Installation instructions
-------------------------

You can install the latest release of CloudDrift using uv, pip, or Conda.
You can also install the latest development (unreleased) version from GitHub.

.. note::

   The ``uv`` instructions below require `uv <https://docs.astral.sh/uv/>`_ to be
   installed. To install it, run:

   .. code-block:: text

     curl -LsSf https://astral.sh/uv/install.sh | sh

   See the `uv installation docs <https://docs.astral.sh/uv/getting-started/installation/>`_
   for other platforms and methods.

uv
^^

In your project, type:

.. code-block:: text

  uv add clouddrift

To install optional dependencies needed by the ``clouddrift.plotting`` module,
type:

.. code-block:: text

  uv add 'clouddrift[plotting]'

For local development, clone the repository and run:

.. code-block:: text

  uv sync

pip
^^^

In your virtual environment, type:

.. code-block:: text

  pip install clouddrift

To install optional dependencies needed by the ``clouddrift.plotting`` module,
type:

.. code-block:: text

  pip install 'clouddrift[plotting]'

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

If you need the latest development version, clone the repository and use uv:

.. code-block:: text

  git clone https://github.com/Cloud-Drift/clouddrift
  cd clouddrift
  uv sync

Alternatively, install directly from GitHub using pip:

.. code-block:: text

  pip install git+https://github.com/Cloud-Drift/clouddrift

Running tests
=============

To run the tests, first clone the repository and install dependencies with uv:

.. code-block:: text

  git clone https://github.com/cloud-drift/clouddrift
  cd clouddrift
  uv sync

Then, run the tests like this:

.. code-block:: text

  uv run pytest

A quick how-to guide is provided on the `Usage <https://cloud-drift.github.io/clouddrift/usage.html>`_ page.

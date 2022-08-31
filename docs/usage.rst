.. _usage:

Usage
=====

Data format
-----------

The first release of CloudDrift provide a easy way to convert any Lagrangian datasets into an archive of ragged arrays. We provide step-by-step guide to convert the trajectories from the Global Drifter Program (GDP) hourly and 6-hourly datasets, an drifters experiment lead by `CARTHE <http://carthe.org/>`_, and a typical numerical Lagrangian output. 

Below is a quick overview to transform an observational Lagrangian dataset stored into multiple files, or a numerical output from a Lagrangian simulation framework. **Detailed examples** are provided as Jupyter Notebooks and can be tested directly in a `Binder <https://mybinder.org/v2/gh/Cloud-Drift/clouddrift/main?labpath=examples>`_ environment.

Collection of files
~~~~~~~~~~~~~~~~~~~

First, to create a ragged arrays archive for a dataset where each trajectory is stored into a individual archive, e.g. the `GDP hourly dataset <https://www.aoml.noaa.gov/phod/gdp/hourly_data.php>`_, it is required to define a `preprocessing` function that returns an `xarray` dataset of a trajectory from its identification number.

.. code-block:: python

   def preprocess(index: int) -> xr.Dataset:
      """
      :param index: drifter's identification number
      :return: xr.Dataset containing the data and attributes
      """
      ds = xr.load_dataset(f'data/file_{index}.nc')

      # perform required preprocessing steps
      # e.g. change units, remove variables, fix attributes, etc.

      return ds

This function will be called for each indices of the dataset (`ids`) to construct the ragged arrays archive, as follow. The ragged arrays contains the required coordinates variables, as well as the specified metadata and data variables. Note that metadata variables contain one value per trajectory while the data variables contain `n` observations per trajectory.

.. code-block:: python

   ids = [1,2,3]  # trajectories to combine

   # mandatory coordinates variables
   coords = {'ids': 'ids', 'time': 'time', 'lon': 'longitude', 'lat': 'latitude'}

   # list of metadata and data from files to include in archive
   metadata = ['ID', 'rowsize']
   data = ['ve', 'vn']

   ra = ragged_array.from_files(ids, preprocess, coords, metadata, data)

which can be easily export to either a parquet,

.. code-block:: python

   ra.to_parquet('data/archive.parquet')

or a NetCDF archive.

.. code-block:: python

   ra.to_parquet('data/archive.nc')

Lagrangian numerical output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a two-dimensional output (`lon`, `lat`, `time`) from a Lagrangian simulation framework (such as `OceanParcels <https://oceanparcels.org/>`_ or `OpenDrift <https://opendrift.github.io/>`_), the ragged arrays archive can be obtained by reshaping the variables to ragged arrays and populating the following dictionaries containing the coordinates, metadata, data, and attributes.

.. code-block:: python

   # initialize dictionaries
   coords = {}
   metadata = {}

   # note that this example dataset does not contain other data than time, lon, lat, and ids 
   # an empty dictionary "data" is initialize anyway
   data = {}

Numerical outputs are usually stored as a 2D matrix (`trajectory`, `time`) filled with `nan` where there is no data. The first step is to identify the finite values and reshape the dataset.

.. code-block:: python

   ds = xr.open_dataset(join(folder, file), decode_times=False)   
   finite_values = np.isfinite(ds['lon'])
   idx_finite = np.where(finite_values)

   # dimension and id of each trajectory
   rowsize = np.bincount(idx_finite[0])
   unique_id = np.unique(idx_finite[0])

   # coordinate variables
   coords["time"] = np.tile(ds.time.data, (ds.dims['traj'],1))[idx_finite]
   coords["lon"] = ds.lon.data[idx_finite]
   coords["lat"] = ds.lat.data[idx_finite]
   coords["ids"] = np.repeat(unique_id, rowsize)

Once performed, we can include extra metadata, such as the size of each trajectory (`rowsize`), and create the ragged arrays archive.

.. code-block:: python

   # metadata   
   metadata = {}
   metadata["rowsize"] = rowsize
   metadata["ID"] = unique_id

   # create the ragged arrays
   ra = ragged_array(coords, metadata, data)
   ra.to_parquet('data/archive.parquet')

Analysis
--------

Once an archive of ragged arrays is created, CloudDrift provides way to read in and convert the data to an Awkward Array object.

.. code-block:: python

   ra = ragged_array.from_parquet('data/archive.parquet')
   ds = ra.to_awkward()

Over the next year, the CloudDrift project will be developing a cloud-ready analysis library to perform typical Lagrangian workflows.

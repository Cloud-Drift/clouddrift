.. _usage:

Usage
=====

The CloudDrift library provides functions for:

* Easy access to cloud-ready Lagrangian ragged-array datasets;
* Common Lagrangian analysis tasks on ragged arrays;
* Adapting custom Lagrangian datasets into ragged arrays.

Let's start by importing the library and accessing a ready-to-use ragged-array
dataset.

Accessing ragged-array Lagrangian datasets
------------------------------------------

We recommend to import the ``clouddrift`` using the ``cd`` shorthand, for convenience:

>>> import clouddrift as cd

CloudDrift provides a set of Lagrangian datasets that are ready to use.
They can be accessed via the ``datasets`` submodule.
In this example, we will load the NOAA's Global Drifter Program (GDP) hourly
dataset, which is hosted in a public AWS bucket as a cloud-optimized Zarr
dataset:

>>> ds = cd.datasets.gdp1h()
>>> ds
<xarray.Dataset>
Dimensions:                (traj: 17324, obs: 165754333)
Coordinates:
    ids                    (obs) int64 ...
    lat                    (obs) float32 ...
    lon                    (obs) float32 ...
    time                   (obs) datetime64[ns] ...
Dimensions without coordinates: traj, obs
Data variables: (12/55)
    BuoyTypeManufacturer   (traj) |S20 ...
    BuoyTypeSensorArray    (traj) |S20 ...
    CurrentProgram         (traj) float64 ...
    DeployingCountry       (traj) |S20 ...
    DeployingShip          (traj) |S20 ...
    DeploymentComments     (traj) |S20 ...
    ...                     ...
    sst1                   (obs) float64 ...
    sst2                   (obs) float64 ...
    typebuoy               (traj) |S10 ...
    typedeath              (traj) int8 ...
    ve                     (obs) float32 ...
    vn                     (obs) float32 ...
Attributes: (12/16)
    Conventions:       CF-1.6
    acknowledgement:   Elipot, Shane; Sykulski, Adam; Lumpkin, Rick; Centurio...
    contributor_name:  NOAA Global Drifter Program
    contributor_role:  Data Acquisition Center
    date_created:      2022-12-09T06:02:29.684949
    doi:               10.25921/x46c-3620
    ...                ...
    processing_level:  Level 2 QC by GDP drifter DAC
    publisher_email:   aoml.dftr@noaa.gov
    publisher_name:    GDP Drifter DAC
    publisher_url:     https://www.aoml.noaa.gov/phod/gdp
    summary:           Global Drifter Program hourly data
    title:             Global Drifter Program hourly drifting buoy collection

The ``gdp1h`` function returns an Xarray ``Dataset`` instance of the ragged-array dataset.
While the dataset is quite large, around a dozen GB, it is not downloaded to your
local machine. Instead, the dataset is accessed directly from the cloud, and only
the data that is needed for the analysis is downloaded. This is possible thanks to
the cloud-optimized Zarr format, which allows for efficient access to the data
stored in the cloud.

Let's look at some variables in this dataset:

>>> ds.lon
<xarray.DataArray 'lon' (obs: 165754333)>
[165754333 values with dtype=float32]
Coordinates:
    ids      (obs) int64 ...
    lat      (obs) float32 ...
    lon      (obs) float32 ...
    time     (obs) datetime64[ns] ...
Dimensions without coordinates: obs
Attributes:
    long_name:  Longitude
    units:      degrees_east

You see that this array is very long--it has 165754333 elements.
This is because in a ragged array, many varying-length arrays are laid out as a
contiguous 1-dimensional array in memory.

Let's look at the dataset dimensions:

>>> ds.dims
Frozen({'traj': 17324, 'obs': 165754333})

The ``traj`` dimension has 17324 elements, which is the number of individual
trajectories in the dataset.
The sum of their lengths equals the length of the ``obs`` dimension.
Internally, these dimensions, their lengths, and the ``count`` (or ``rowsize``)
variables are used internally to make CloudDrift's analysis functions aware of
the bounds of each contiguous array within the ragged-array data structure.

Doing common analysis tasks on ragged arrays
--------------------------------------------

Now that we have a ragged-array dataset loaded as an Xarray ``Dataset`` instance,
let's do some common analysis tasks on it.

TODO

Adapting custom Lagrangian datasets into ragged arrays
------------------------------------------------------

CloudDrift provides an easy way to convert custom Lagrangian datasets into
`contiguous ragged arrays <https://cfconventions.org/cf-conventions/cf-conventions.html#_contiguous_ragged_array_representation>`_.

.. code-block:: python

    # Import a GDP-hourly adapter function
    from clouddrift.adapters.gdp import to_raggedarray

    # Download 100 random GDP-hourly trajectories as a ragged array
    ra = to_raggedarray(n_random_id=100)

    # Store to NetCDF and Parquet files
    ra.to_netcdf("gdp.nc")
    ra.to_parquet("gdp.parquet")

    # Convert to Xarray Dataset for analysis
    ds = ra.to_xarray()

    # Alternatively, convert to Awkward Array for analysis
    ds = ra.to_awkward()

This snippet is specific to the hourly GDP dataset, however, you can use the
``RaggedArray`` class directly to convert other custom datasets into a ragged
array structure that is analysis ready via Xarray or Awkward Array packages.
The functions to do that are defined in the ``clouddrift.adapters`` submodule.
You can use these examples as a reference to ingest your own or other custom
Lagrangian datasets into ``RaggedArray``.
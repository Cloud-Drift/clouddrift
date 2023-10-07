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
Internally, these dimensions, their lengths, and the ``rowsize``
variable are used internally to make CloudDrift's analysis functions aware of
the bounds of each contiguous array within the ragged-array data structure.

Doing common analysis tasks on ragged arrays
--------------------------------------------

Now that we have a ragged-array dataset loaded as an Xarray ``Dataset`` instance,
let's do some common analysis tasks on it.
Our dataset is on a remote server and fairly large (a dozen GB or so), so let's
first subset it to several trajectories so that we can more easily work with it.
The variable ``ID`` is the unique identifier for each trajectory:

>>> ds.ID[:10].values
array([2578, 2582, 2583, 2592, 2612, 2613, 2622, 2623, 2931, 2932])

>>> from clouddrift.ragged import subset

``subset`` allows you to subset a ragged array by some criterion.
In this case, we will subset it by the ``ID`` variable:

>>> ds_sub = subset(ds, {"ID": list(ds.ID[:5])})
>>> ds_sub
<xarray.Dataset>
Dimensions:                (traj: 5, obs: 13612)
Coordinates:
    ids                    (obs) int64 2578 2578 2578 2578 ... 2612 2612 2612
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

You see that we now have a subset of the original dataset, with 5 trajectories
and a total of 13612 observations.
This subset is small enough to quickly and easily work with for demonstration
purposes.
Let's see how we can compute the mean and maximum velocities of each trajectory.
To start, we'll need to obtain the velocities over all trajectory times.
Although the GDP dataset already comes with velocity variables, we won't use
them here so that we can learn how to compute them ourselves from positions.
``clouddrift``'s ``kinematics`` module provides the ``velocity_from_position``
function that allows you to do just that.

>>> from clouddrift.kinematics import velocity_from_position

At a minimum ``velocity_from_position`` requires three input parameters:
consecutive x- and y-coordinates and time, so we could do:

>>> u, v = velocity_from_position(ds_sub.lon, ds_sub.lat, ds_sub.time)

``velocity_from_position`` returns two arrays, ``u`` and ``v``, which are the
zonal and meridional velocities, respectively.
By default, it assumes that the coordinates are in degrees, and it handles the
great circle path calculation and longitude wraparound under the hood.
However, recall that ``ds_sub.lon``, ``ds_sub.lat``, and ``ds_sub.time`` are
ragged arrays, so we need a different approach to calculate velocities while
respecting the trajectory boundaries.
For this, we can use the ``ragged_apply`` function, which applies a function
to each trajectory in a ragged array, and returns the concatenated result.

>>> from clouddrift.ragged import apply_ragged
>>> u, v = apply_ragged(velocity_from_position, [ds_sub.lon, ds_sub.lat, ds_sub.time], ds_sub.rowsize)

``u`` and ``v`` here are still ragged arrays, which means that the five
contiguous trajectories are concatenated into 1-dimensional arrays.

Now, let's compute the velocity magnitude in meters per second.
The time in this dataset is loaded in nanoseconds by default:

>>> ds_sub.time.values
array(['2005-04-15T20:00:00.000000000', '2005-04-15T21:00:00.000000000',
       '2005-04-15T22:00:00.000000000', ...,
       '2005-10-02T03:00:00.000000000', '2005-10-02T04:00:00.000000000',
       '2005-10-02T05:00:00.000000000'], dtype='datetime64[ns]')

So, to obtain the velocity magnitude in meters per second, we'll need to
multiply our velocities by ``1e9``.

>>> velocity_magnitude = np.sqrt(u**2 + v**2) * 1e9
>>> velocity_magnitude
array([0.28053388, 0.6164632 , 0.89032112, ..., 0.2790803 , 0.20095603,
       0.20095603])

>>> velocity_magnitude.mean(), velocity_magnitude.max()
(0.22115242718877506, 1.6958275672626286)

However, these aren't the results we are looking for! Recall that we have the
velocity magnitude of five different trajectories concatenated into one array.
This means that we need to use ``apply_ragged`` again to compute the mean and
maximum values:

>>> apply_ragged(np.mean, [velocity_magnitude], ds_sub.rowsize)
array([0.32865148, 0.17752435, 0.1220523 , 0.13281067, 0.14041268])
>>> apply_ragged(np.max, [velocity_magnitude], ds_sub.rowsize)
array([1.69582757, 1.36804354, 0.97343434, 0.60353528, 1.05044213])

And there you go! We used ``clouddrift`` to:

#. Load a real-world Lagrangian dataset from the cloud;
#. Subset the dataset by trajectory IDs;
#. Compute the velocity vectors and their magnitudes for each trajectory;
#. Compute the mean and maximum velocity magnitudes for each trajectory.

``clouddrift`` offers many more functions for common Lagrangian analysis tasks.
Please explore the `API <https://cloud-drift.github.io/clouddrift/api.html>`_
to learn about other functions and how to use them.

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
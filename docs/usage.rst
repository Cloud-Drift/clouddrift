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
<xarray.Dataset> Size: 16GB
Dimensions:                (traj: 19396, obs: 197214787)
Coordinates:
    id                     (traj) int64 155kB ...
    time                   (obs) datetime64[ns] 2GB ...
Dimensions without coordinates: traj, obs
Data variables: (12/59)
    BuoyTypeManufacturer   (traj) |S20 388kB ...
    BuoyTypeSensorArray    (traj) |S20 388kB ...
    CurrentProgram         (traj) float32 78kB ...
    DeployingCountry       (traj) |S20 388kB ...
    DeployingShip          (traj) |S20 388kB ...
    DeploymentComments     (traj) |S20 388kB ...
    ...                     ...
    start_lat              (traj) float32 78kB ...
    start_lon              (traj) float32 78kB ...
    typebuoy               (traj) |S10 194kB ...
    typedeath              (traj) int8 19kB ...
    ve                     (obs) float32 789MB ...
    vn                     (obs) float32 789MB ...
Attributes: (12/16)
    Conventions:       CF-1.6
    acknowledgement:   Elipot, Shane; Sykulski, Adam; Lumpkin, Rick; Centurio...
    contributor_name:  NOAA Global Drifter Program
    contributor_role:  Data Acquisition Center
    date_created:      2023-09-08T17:05:12.130123
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
<xarray.DataArray 'lon' (obs: 197214787)> Size: 2GB
[197214787 values with dtype=float64]
Coordinates:
    time     (obs) datetime64[ns] 2GB ...
Dimensions without coordinates: obs
Attributes:
    long_name:  Longitude
    units:      degrees_east

You see that this array is very long--it has 197214787 elements.
This is because in a ragged array, many varying-length arrays are laid out as a
contiguous 1-dimensional array in memory.

Let's look at the dataset dimensions:

>>> ds.sizes
Frozen({'traj': 19396, 'obs': 197214787})

The ``traj`` dimension has 19396 elements, which is the number of individual
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
The variable ``id`` is the unique identifier for each trajectory:

>>> ds.id[:10].values
array([8707978, 8707988, 8707928, 8707971, 8911585, 8911583, 8911512,
       8911524, 8911517, 8911521])

>>> from clouddrift.ragged import subset

``subset`` allows you to subset a ragged array by some criterion.
In this case, we will subset it by the ``id`` variable:

>>> ds_sub = ds_sub = subset(ds, {"id": list(ds.id[:5])}, row_dim_name="traj")
>>> ds_sub
<xarray.Dataset> Size: 849kB
Dimensions:                (traj: 5, obs: 10595)
Coordinates:
    id                     (traj) int64 40B 8707978 8707988 ... 8707971 8911585
    time                   (obs) datetime64[ns] 85kB ...
Dimensions without coordinates: traj, obs
Data variables: (12/59)
    BuoyTypeManufacturer   (traj) |S20 100B ...
    BuoyTypeSensorArray    (traj) |S20 100B ...
    CurrentProgram         (traj) float32 20B ...
    DeployingCountry       (traj) |S20 100B ...
    DeployingShip          (traj) |S20 100B ...
    DeploymentComments     (traj) |S20 100B ...
    ...                     ...
    start_lat              (traj) float32 20B ...
    start_lon              (traj) float32 20B ...
    typebuoy               (traj) |S10 50B ...
    typedeath              (traj) int8 5B ...
    ve                     (obs) float32 42kB ...
    vn                     (obs) float32 42kB ...
Attributes: (12/16)
    Conventions:       CF-1.6
    acknowledgement:   Elipot, Shane; Sykulski, Adam; Lumpkin, Rick; Centurio...
    contributor_name:  NOAA Global Drifter Program
    contributor_role:  Data Acquisition Center
    date_created:      2023-09-08T17:05:12.130123
    doi:               10.25921/x46c-3620
    ...                ...
    processing_level:  Level 2 QC by GDP drifter DAC
    publisher_email:   aoml.dftr@noaa.gov
    publisher_name:    GDP Drifter DAC
    publisher_url:     https://www.aoml.noaa.gov/phod/gdp
    summary:           Global Drifter Program hourly data
    title:             Global Drifter Program hourly drifting buoy collection

You see that we now have a subset of the original dataset, with 5 trajectories
and a total of 10595 observations.
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
array(['1987-10-02T13:00:00.000000000', '1987-10-02T14:00:00.000000000',
       '1987-10-02T15:00:00.000000000', ...,
       '1990-04-18T13:00:00.000000000', '1990-04-18T14:00:00.000000000',
       '1990-04-18T15:00:00.000000000'], dtype='datetime64[ns]')

So, to obtain the velocity magnitude in meters per second, we'll need to
multiply our velocities by ``1e9``.

>>> import numpy as np
>>> velocity_magnitude = np.sqrt(u**2 + v**2) * 1e9
>>> velocity_magnitude
array([0.38879039, 0.31584365, 0.3195847 , ..., 0.33568719, 0.90269212,
       0.90269212])

>>> velocity_magnitude.mean(), velocity_magnitude.max()
(0.23904881731647423, 3.5152917225763334)

However, these aren't the results we are looking for! Recall that we have the
velocity magnitude of five different trajectories concatenated into one array.
This means that we need to use ``apply_ragged`` again to compute the mean and
maximum values:

>>> apply_ragged(np.mean, [velocity_magnitude], ds_sub.rowsize)
array([0.20233901, 0.22178923, 0.35035411, 0.3076482 , 0.20912337])
>>> apply_ragged(np.max, [velocity_magnitude], ds_sub.rowsize)
array([1.31215785, 1.42468528, 2.26800085, 2.14766676, 3.51529172])

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
    from clouddrift.adapters.gdp1h import to_raggedarray

    # Download 100 random GDP-hourly trajectories as a ragged array
    ra = to_raggedarray(n_random_id=100)

    # Store to NetCDF and Parquet files
    ra.to_netcdf("gdp.nc")
    ra.to_parquet("gdp.parquet")

    # Convert to Xarray Dataset for analysis
    ds = ra.to_xarray()

    # Alternatively, convert to Awkward Array for analysis
    ds = ra.to_awkward()

The snippet above is specific to the hourly GDP dataset, however, you can use the
``RaggedArray`` class directly to convert other custom datasets into a ragged
array structure that is analysis ready via Xarray or Awkward Array packages.
The functions to do that are defined in the ``clouddrift.adapters`` submodule.
You can use these examples as a reference to ingest your own or other custom
Lagrangian datasets into ``RaggedArray``. We provide below a simple example on 
how to build ragged array datasets from simulated data:

.. code-block:: python

    # create a synthetic Lagrangian data of 8 trajectories of random walks
    import numpy as np

    rowsize = [100,65,7,22,56,78,99,70]
    x = []
    y = []
    for i in range(len(rowsize)):
        x.append(np.cumsum(np.random.normal(0,1,rowsize[i]) + np.random.uniform(0,1,1)))
        y.append(np.cumsum(np.random.normal(0,1,rowsize[i]) + np.random.uniform(0,1,1)))
    x = np.concatenate(x)
    y = np.concatenate(y)

    # create an instance of the RaggedArray class
    from clouddrift.raggedarray import RaggedArray

    # define the actual coordinates
    coords = {"time": np.arange(len(x)), "id": np.arange(len(rowsize))}
    # define the data
    data = {"x": x, "y": y}
    # define the metadata which here include the `rowsize` parameter
    metadata = {"rowsize": rowsize}

    # map the names of the dimensions to what the class expects, that is 
    # what are the names of "rows" and "obs"
    name_dims = {"traj": "rows", "obs": "obs"}
    # map the dimensions of the coordinates defined above 
    coord_dims = {"time": "obs", "id": "traj"}
    # define some attributes for the dataset and its variables
    attrs_global = {"title":"An example of synthetic data"}
    attrs_variables = {"id" : {"long_name": "trajectory id"},
                    "time": {"long_name": "time"},
                    "x": {"long_name": "x coordinate"},
                    "y": {"long_name": "y coordinate"},
                    "rowsize": {"long_name": "number of observations in each trajectory"}}
    # instantiate the RaggedArray class
    ra = RaggedArray(
        coords, metadata, data, attrs_global, attrs_variables, name_dims, coord_dims
    )

Next the ragged array object ``ra`` can be used to generate xarray and awkward datasets for further analysis and processing:

.. code-block:: python

    # convert to xarray dataset
    ds = ra.to_xarray()
    ds
    <xarray.Dataset> Size: 12kB
    Dimensions:  (traj: 8, obs: 497)
    Coordinates:
        time     (obs) int64 4kB 0 1 2 3 4 5 6 7 ... 489 490 491 492 493 494 495 496
        id       (traj) int64 64B 0 1 2 3 4 5 6 7
    Dimensions without coordinates: traj, obs
    Data variables:
        rowsize  (traj) int64 64B 100 65 7 22 56 78 99 70
        x        (obs) float64 4kB -0.3243 -0.2817 0.1442 1.31 ... 13.18 13.07 14.02
        y        (obs) float64 4kB 1.25 2.073 3.493 3.44 ... 11.56 9.913 10.11 11.03
    Attributes:
        title:    An example of synthetic data

    # convert to awkward array
    ds_ak = ra.to_awkward()
    ds_ak
    [{rowsize: 100, obs: {time: [...], ...}},
    {rowsize: 65, obs: {time: [...], id: 1, ...}},
    {rowsize: 7, obs: {time: [...], id: 2, ...}},
    {rowsize: 22, obs: {time: [...], id: 3, ...}},
    {rowsize: 56, obs: {time: [...], id: 4, ...}},
    {rowsize: 78, obs: {time: [...], id: 5, ...}},
    {rowsize: 99, obs: {time: [...], id: 6, ...}},
    {rowsize: 70, obs: {time: [...], id: 7, ...}}]
    -----------------------------------------------------------------------------------------------------
    type: 8 * struct[{
        rowsize: int64[parameters={"attrs": {"long_name": "number of observations in each trajectory"}}],
        obs: {
            time: [var * int64, parameters={"attrs": {"long_name": "time"}}],
            id: int64[parameters={"attrs": {"long_name": "trajectory id"}}],
            x: [var * float64, parameters={"attrs": {"long_name": "x coordinate"}}],
            y: [var * float64, parameters={"attrs": {"long_name": "y coordinate"}}]
        }
    }, parameters={"attrs": {"title": "An example of synthetic data"}}]
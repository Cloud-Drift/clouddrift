.. _usage:

Usage
=====

CloudDrift provides an easy way to convert Lagrangian datasets into
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
We provide step-by-step guides to convert the individual trajectories from the
Global Drifter Program (GDP) hourly and 6-hourly datasets, the drifters from the
`CARTHE <http://carthe.org/>`_ experiment, and a typical output from a numerical
Lagrangian experiment in our
`repository of example Jupyter Notebooks <https://github.com/cloud-drift/clouddrift-examples>`_.
You can use these examples as a reference to ingest your own or other custom
Lagrangian datasets into ``RaggedArray``.

In the future, ``clouddrift`` will be including functions to perform typical
oceanographic Lagrangian analyses.

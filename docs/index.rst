CloudDrift, a platform for accelerating research with Lagrangian climate data
=============================================================================

Version: |version|
----------------------

Lagrangian data typically refers to oceanic and atmosphere information acquired by observing platforms drifting with the flow they are embedded within, but also refers more broadly to the data originating from uncrewed platforms, vehicles, and animals that gather data along their unrestricted and often complex paths. Because such paths traverse both spatial and temporal dimensions, Lagrangian data can convolve spatial and temporal information that cannot always readily be organized in common data structures and stored in standard file formats with the help of common libraries and standards.

As such, for both originators and users, Lagrangian data present challenges that the CloudDrift project aims to overcome. This project is funded by the `NSF EarthCube program <https://www.earthcube.org/info>`_ through `EarthCube Capabilities Grant No. 2126413 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=2126413>`_.

Scope and Key Features
----------------------

The scope of the Clouddrift library includes: 

1. **Working with contiguous ragged array representations of data, wether they originate from geosciences or any other field**. Ragged array representations are typically useful when the data lengths of the instances of a feature (variable) are not all equal. With such representations the data for each features are stored contiguously in memory, and the number of elements that each feature has is contained in a count variable which Clouddrift calls *rowsize*. 

2. **Delivering functions and methods to perform scientific analysis of Lagrangian data, oceanographic or otherwise, structured as ragged arrays or otherwise**. A straightforward example of Lagrangian analysis provided by Clouddrift is the derivation of Lagrangian velocities from a sequence of Lagrangian positions, and vice versa. Another more involved example is the discovery of pairs of Lagrangian data prescribed by distances in space and time. Both of these methods are currently available with Clouddrift.

3. **Processing publicly available Lagrangian datasets into the common ragged array data structure and format**. Through data *adapters*, this type of processing includes not only converting Lagrangian data from typically regular arrays to ragged arrays but also aggregating data and metadata from multiple data files into a single data file. The canonical example of the Clouddrift library is constituted of the data from the NOAA Global Drifter Program (see Motivations below).

4. **Making cloud-optimized ragged array datasets easily accessible**. This involves opening in a local computing environment, without unnecessary download, Lagrangian datasets available from cloud servers, as well as opening Lagrangian dataset which have been seamlessly processed by the Clouddrift data *adapters*.    

CloudDrift's analysis functions are principally centered around the ragged-array data structure:

.. image:: img/ragged_array.png
  :width: 800
  :align: center
  :alt: Ragged array schematic

Motivations
-----------

The `Global Drifter Program (GDP) <https://www.aoml.noaa.gov/phod/gdp/>`_ of the US National Oceanic and Atmospheric Administration has released to date nearly 25,000 drifting buoys, or drifters, with the goal of obtaining observations of oceanic velocity, sea surface temperature, and sea level pressure. From these drifter observations, the GDP generates two data products: one of oceanic variables estimated along drifter trajectories at `hourly <https://www.aoml.noaa.gov/phod/gdp/interpolated/data/all.php>`_ time steps, and one at `six-hourly <https://www.aoml.noaa.gov/phod/gdp/hourly_data.php>`_ steps.

There are a few ways to retrieve the data, but all typically require time-consuming preprocessing steps in order to prepare the data for analysis. As an example, the datasets can be retrieved through an `ERDDAP server <https://data.pmel.noaa.gov/generic/erddap/tabledap/gdp_hourly_velocities.html>`_, but requests are limited in size. The latest `6-hourly dataset <https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/netcdf/>`_ is distributed as a collection of thousands of individual NetCDF files or as a series of `ASCII files <https://www.aoml.noaa.gov/phod/gdp/>`_. Until recently, the `hourly dataset <https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/v2.00/netcdf/>`_ was distributed as a collection of individual NetCDF files (17,324 for version 1.04c) but is now distributed by NOAA NCEI as a `single NetCDF file <https://doi.org/10.25921/x46c-3620>`_ containing a series of ragged arrays, thanks to the work of CloudDrift. A single file simplifies data distribution, decreases metadata redundancies, and efficiently stores a Lagrangian data collection of uneven lengths.

Getting started
---------------

* :doc:`install`
* :doc:`usage`
* :doc:`datasets`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   install
   usage
   datasets

Reference
---------

* :doc:`api`
* contributing_

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Reference

    Contributing <https://github.com/Cloud-Drift/clouddrift/blob/main/CONTRIBUTING.md>
    api

.. _contributing: https://github.com/Cloud-Drift/clouddrift/blob/main/CONTRIBUTING.md
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

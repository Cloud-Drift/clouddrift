CloudDrift, accelerating Lagrangian Analysis
============================================

For data, Lagrangian refers to oceanic and atmosphere information acquired by observing platforms drifting with the flow they are embedded within, but also refers more broadly to the data originating from uncrewed platforms, vehicles, and animals that gather data along their unrestricted and often complex paths. Because such paths traverse both spatial and temporal dimensions, Lagrangian data often convolve spatial and temporal information that cannot always readily be organized in common data structures and stored in standard file formats with the help of common libraries and standards. 

As such, for both originators and users, Lagrangian data present challenges that the EarthCube CloudDrift project aims to overcome.

Motivation
----------

As of today, 24,597 drifters were released as part of the Global Drifter Program (GDP) dataset. From those observations, the GDP provides interpolated `hourly <https://www.aoml.noaa.gov/phod/gdp/interpolated/data/all.php>`_ and `six-hourly <https://www.aoml.noaa.gov/phod/gdp/hourly_data.php>`_ datasets.

There are a few ways to retrieve the data, but all required time-consuming preprocessing performed steps. The 6-hourly dataset is distributed as a serie of `ASCII files <https://www.aoml.noaa.gov/phod/gdp/>`_ or a collections of 24,597 `NetCDF files <https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/netcdf/>`_, while the hourly dataset is distributed as a collection of 17,324 `NetCDF files <https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/v2.00/netcdf/>`_. Alternatively, the dataset can be retrieved through the `ERDDAP server <https://data.pmel.noaa.gov/generic/erddap/tabledap/gdp_hourly_velocities.html>`_, but requests are limited in size. 

CloudDrift goals are to *simplify* the necessary steps to get started with the GDP dataset, and to provide a cloud-ready library to *accelerate* Lagrangian analysis.

Getting started
---------------

* :doc:`install`
* :doc:`usage`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   install
   usage

Reference
---------

* :doc:`contributing`
* :doc:`api`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Reference

    contributing
    api

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

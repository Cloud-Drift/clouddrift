.. _why:

Why CloudDrift ?
================

The Global Drifter Program (GDP) dataset combines currently the data of 24,597 drifters from February 1979 to June 2020. Both the 6-hourly and the hourly datasets are not critically large at roughly 2 GB and 10 GB, respectively.
 
As of today, there are a few ways to retrieve the data, but all required some preprocessing performed by the users. The 6-hourly dataset is distributed as a serie of `ASCII files <https://www.aoml.noaa.gov/phod/gdp/>`_, while the hourly dataset is distributed as a collection of 17,324 `netCDF files <https://www.aoml.noaa.gov/phod/gdp/hourly_data.php>`_. Subset of the dataset can also be retrieved through an `ERDDAP server <https://data.pmel.noaa.gov/generic/erddap/tabledap/gdp_hourly_velocities.html>`_, but requests are limited in size. 

CloudDrift goals are to *simplify* the necessary steps to get started with the GDP dataset, and to provide a cloud-ready library to *accelerate* Lagrangian analysis.

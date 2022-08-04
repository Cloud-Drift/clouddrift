
# Lagrangian Data

The *CloudDrift* library operates on Lagrangian data that are formatted in ragged arrays. A ragged array (also known as [jagged array](https://en.wikipedia.org/wiki/Jagged_array)), is *an array of arrays of which the member arrays can be of different lengths, producing rows of jagged edges when visualized as output*.

Because Lagrangian data do not typically come as a ragged array, we have built tools to convert various Lagrangian datasets into this data format within the python computing environment. Note that data organized in this way can then be exported into language and platform-independent files such as NetCDF or Parquet.

## Structure

The `data/` folder contains modules (e.g. `gdp.py`) that can be used to convert raw data contained in the `raw/` subfolder (e.g. `raw/gdp-v2.00/`) and written in a processed file in the `process/` subfolder (e.g. `process/gdp_v2.00.nc`).

Taking as an example the collection of surface drifter data from the [NOAA Global Drifter Program](https://www.aoml.noaa.gov/phod/gdp/) (GDP). The data from that program come in two products that can be downloaded in a number of ways. With the tools described below and created as part of *CloudDrift*, the high resolution hourly data product ([link](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php)) is now officially distributed by [NOAA NCEI](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:AOML-GDP-1HR) as a single ragged array NetCDF file ([link](https://www.nodc.noaa.gov/archive/arc0199/0248584/1.1/data/0-data/gdp_v2.00.nc)).

## Raw data

Raw data must be available in the appropriate `raw/` subfolder before a dataset is processed.

For the GDP dataset (and also other available *examples*), a download function is provided (e.g. `gdp.download()`) to automatically retrieve any of the 17,324 individual NetCDF files from the AOML repository and stored them in the appropriate `raw/` subfolder.

Alternatively, the GDP hourly data can be downloaded via the HTTPS ([link](https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/v2.00/netcdf/)) or FTP ([link](ftp://ftp.aoml.noaa.gov/phod/pub/lumpkin/hourly/v2.00/netcdf/)) repository of the Data Assembly Center at NOAA AOML. The scripts `sync_gdp_hourly.sh` (and `sync_gdp_6hourly.sh`) can be used to automatically retrieve the latest dataset (or update a local dataset). Both require the `lftp` command, which is available on most Linux Distribution or through for macOS users ([link](https://formulae.brew.sh/formula/lftp)).

## Process data

Processed data as ragged arrays are stored in the appropriate `process/` subfolder.

For the GDP datasets, we suggest retrieving directly the ragged array NetCDF file `gdp_v2.00.nc` from NOAA NCEI ([link](https://www.nodc.noaa.gov/archive/arc0199/0248584/1.1/data/0-data/gdp_v2.00.nc)) and stored it in the `process/` folder (or any convenient location).

For *advanced users* and *data originators*, Notebooks are made available to present the steps to manually create a NetCDF ragged array from a collection of individual trajectory files (e.g. `dataformat-gdp.ipynb`) or the numerical output of a Lagrangian simulation (e.g. `dataformat-numerical.ipynb`).

## References

- Elipot, S., R. Lumpkin, R. C. Perez, J. M. Lilly, J. J. Early, and A. M. Sykulski (2016), "A global surface drifter dataset at hourly resolution", J. Geophys. Res. Oceans, 121, [doi:10.1002/2016JC011716](doi:10.1002/2016JC011716) [(Archived pdf)](https://www.aoml.noaa.gov/phod/gdp/papers/Elipot_et_al-2016.pdf)

- Elipot, Shane, Adam Sykulski, Rick Lumpkin, Luca Centurioni, and Mayra Pazos (2021), *A Dataset of Hourly Sea Surface Temperature From Drifting Buoys*, in revision for [Scientific Data](https://www.nature.com/sdata/). Preprint available from arXiv as [arxiv:2201.08289v1](https://arxiv.org/abs/2201.08289v1).

- Elipot, Shane; Sykulski, Adam; Lumpkin, Rick; Centurioni, Luca; Pazos, Mayra (2022). *Hourly location, current velocity, and temperature collected from Global Drifter Program drifters world-wide*. [indicate subset used]. NOAA National Centers for Environmental Information. Dataset. [https://doi.org/10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620). Accessed [date].

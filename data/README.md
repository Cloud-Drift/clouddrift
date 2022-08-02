
The CloudDrift library typically operates on Lagrangian data that are formatted in ragged arrays. A ragged array (also known as [jagged array](https://en.wikipedia.org/wiki/Jagged_array)), is *an array of arrays of which the member arrays can be of different lengths, producing rows of jagged edges when visualized as output*.

Because Lagrangian data do not typically come as ragged array, we have built tools to convert various Lagrangian datasets into this data format within the python computing environment. Note that data organized in this way can then be written into language and platform-independent files such as NetCDF or Parquet.

The `data/` folder contains code (e.g. `gdp.py`) that can be used to convert raw data contained in the `raw/` subfolder (e.g. `raw/gdp-v2.00/`) and written in a file as processed data in the `process/` subfolder (e.g. `process/gdp_v2.00.nc`).

Taking as an example the collection of surface drifter data from the [NOAA Global Drifter Program](https://www.aoml.noaa.gov/phod/gdp/) (GDP). The data from that program come in two products that can be downloaded in [a number of ways](https://www.aoml.noaa.gov/phod/gdp/data.php). We take the example of the high resolution [hourly data product](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php) which is now officially distributed by [NOAA NCEI](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:AOML-GDP-1HR) as a single file ([direct link to single NetCDF file](https://www.nodc.noaa.gov/archive/arc0199/0248584/1.1/data/0-data/gdp_v2.00.nc)) containing the data as a ragged array. This single file has been created with the tools described below and created as part of CloudDrift.  

## Process data

Processed data as ragged arrays are stored in the `process/` folder.

For *most users*, the NetCDF file `gdp_v2.00.nc`, can be retrieved directly from NOAA NCEI ([direct link to single NetCDF file](https://www.nodc.noaa.gov/archive/arc0199/0248584/1.1/data/0-data/gdp_v2.00.nc)) and stored in the `process/` folder (or any convenient location).

For *advanced users*, the preprocessing routines are made available (`preprocess.ipynb`) to manually create the NetCDF ragged array (`gdp_v2.00.nc`) from the individual trajectory files. This requires downloading the raw dataset as explained below.

## Raw data

The GDP hourly data can alternatively be downloaded individually for each drifter via the [FTP server](ftp://ftp.aoml.noaa.gov/phod/pub/lumpkin/hourly/v2.00/netcdf/) of the Data Assembly Center at NOAA AOML.

### Manually
The [FTP folder](ftp://ftp.aoml.noaa.gov/phod/pub/lumpkin/hourly/v2.00/netcdf/) contains 17,324 netCDF files (for versions 1.04c and 2.00), with the format `drifter_{id}.nc`. Those files have to be downloaded, and extracted in the `raw/` folder before performing the preprocessing.

### Automatically
Alternatively, the script `sync_gdp_hourly.sh` can be used to automatically retrieve (or update) the current dataset. It requires the `lftp` command, which is available on most Linux Distribution or through [Homebrew](https://formulae.brew.sh/formula/lftp) for macOS users.

## References
- Elipot, S., R. Lumpkin, R. C. Perez, J. M. Lilly, J. J. Early, and A. M. Sykulski (2016), "A global surface drifter dataset at hourly resolution", J. Geophys. Res. Oceans, 121, [doi:10.1002/2016JC011716](doi:10.1002/2016JC011716) [(Archived pdf)](https://www.aoml.noaa.gov/phod/gdp/papers/Elipot_et_al-2016.pdf)

- Elipot, Shane, Adam Sykulski, Rick Lumpkin, Luca Centurioni, and Mayra Pazos (2021), *A Dataset of Hourly Sea Surface Temperature From Drifting Buoys*, in revision for [Scientific Data](https://www.nature.com/sdata/). Preprint available from arXiv as [arxiv:2201.08289v1](https://arxiv.org/abs/2201.08289v1).

- Elipot, Shane; Sykulski, Adam; Lumpkin, Rick; Centurioni, Luca; Pazos, Mayra (2022). *Hourly location, current velocity, and temperature collected from Global Drifter Program drifters world-wide*. [indicate subset used]. NOAA National Centers for Environmental Information. Dataset. [https://doi.org/10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620). Accessed [date].

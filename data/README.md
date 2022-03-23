# Process data
To use the CloudDrift library in a local environment, the library requires a ragged array (stored in the `process/` folder) that regroups the individual drifter trajectories.

For *most users*, the netCDF dataset `gdp_v2.00.nc`, can be retrieved from cloud storage ([FTP](ftp://ftp.aoml.noaa.gov/phod/pub/lumpkin/hourly/v2.00) and [soon web]()) and stored in the `process/` folder (or any convenient location).

For *advanced users*, the preprocessing routines are made available (`preprocess.ipynb`) to manually create the netCDF ragged array (`gdp_v2.00.nc`) from the individual trajectory files. This requires downloading the raw dataset which is explained below.

# Raw data
The raw dataset of the [Hourly Drifter Data](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php) is composed of 17,324 netCDF (Nov 3rd, 2021) and is located on the AOML [FTP server](ftp://ftp.aoml.noaa.gov/phod/pub/lumpkin/hourly/v2.00/netcdf/).

### Manually
The FTP folder contains 17,324 netCDF files, with the format `drifter_{id}.nc`. Those files have to be downloaded, and extracted in the `raw/` folder before performing the preprocessing.

### Automatically
Alternatively, the script *sync_gdp_hourly.sh* can be used to automatically retrieve (or update) the current dataset. It requires the `lftp` command, which is available on most Linux Distribution or through [Homebrew](https://formulae.brew.sh/formula/lftp) for macOS users.

## Reference
Elipot, S., R. Lumpkin, R. C. Perez, J. M. Lilly, J. J. Early, and A. M. Sykulski (2016), "A global surface drifter dataset at hourly resolution", J. Geophys. Res. Oceans, 121, [doi:10.1002/2016JC011716](doi:10.1002/2016JC011716) [(Archived pdf)](https://www.aoml.noaa.gov/phod/gdp/papers/Elipot_et_al-2016.pdf)

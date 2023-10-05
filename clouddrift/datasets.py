"""
This module provides functions to easily access ragged-array datasets.
"""

from clouddrift import adapters
import os
import xarray as xr


def gdp1h() -> xr.Dataset:
    """Returns the latest version of the NOAA Global Drifter Program (GDP) hourly
    dataset as an Xarray dataset.

    The data is accessed from zarr archive hosted on a public AWS S3 bucket accessible at
    https://registry.opendata.aws/noaa-oar-hourly-gdp/. Original data source from NOAA NCEI
    is https://doi.org/10.25921/x46c-3620).

    Returns
    -------
    xarray.Dataset
        Hourly GDP dataset as a ragged array

    Examples
    --------
    >>> from clouddrift.datasets import gdp1h
    >>> ds = gdp1h()
    >>> ds
    <xarray.Dataset>
    Dimensions:                (traj: 19396, obs: 197214787)
    Coordinates:
        ids                    (obs) int64 ...
        time                   (obs) datetime64[ns] ...
    Dimensions without coordinates: traj, obs
    Data variables: (12/60)
        BuoyTypeManufacturer   (traj) |S20 ...
        BuoyTypeSensorArray    (traj) |S20 ...
        CurrentProgram         (traj) float32 ...
        DeployingCountry       (traj) |S20 ...
        DeployingShip          (traj) |S20 ...
        DeploymentComments     (traj) |S20 ...
        ...                     ...
        start_lat              (traj) float32 ...
        start_lon              (traj) float32 ...
        typebuoy               (traj) |S10 ...
        typedeath              (traj) int8 ...
        ve                     (obs) float32 ...
        vn                     (obs) float32 ...
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

    See Also
    --------
    :func:`gdp6h`
    """
    url = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp-v2.01.zarr"
    return xr.open_dataset(url, engine="zarr")


def gdp6h() -> xr.Dataset:
    """Returns the NOAA Global Drifter Program (GDP) 6-hourly dataset as an
    Xarray dataset.

    The data is accessed from a public HTTPS server at NOAA's Atlantic
    Oceanographic and Meteorological Laboratory (AOML) accessible at
    https://www.aoml.noaa.gov/phod/gdp/index.php.

    Returns
    -------
    xarray.Dataset
        6-hourly GDP dataset as a ragged array

    Examples
    --------
    >>> from clouddrift.datasets import gdp6h
    >>> ds = gdp6h()
    >>> ds
    <xarray.Dataset>
    Dimensions:                (traj: 26843, obs: 44544647)
    Coordinates:
        ids                    (obs) int64 ...
        time                   (obs) datetime64[ns] ...
        lon                    (obs) float32 ...
        lat                    (obs) float32 ...
    Dimensions without coordinates: traj, obs
    Data variables: (12/44)
        ID                     (traj) int64 ...
        rowsize                (traj) int32 ...
        WMO                    (traj) int32 ...
        expno                  (traj) int32 ...
        deploy_date            (traj) datetime64[ns] ...
        deploy_lat             (traj) float32 ...
        ...                     ...
        vn                     (obs) float32 ...
        temp                   (obs) float32 ...
        err_lat                (obs) float32 ...
        err_lon                (obs) float32 ...
        err_temp               (obs) float32 ...
        drogue_status          (obs) bool ...
    Attributes: (12/16)
        title:             Global Drifter Program six-hourly drifting buoy collec...
        history:           Last update July 2022.  Metadata from dirall.dat and d...
        Conventions:       CF-1.6
        date_created:      2022-12-08T18:44:27.784441
        publisher_name:    GDP Drifter DAC
        publisher_email:   aoml.dftr@noaa.gov
        ...                ...
        contributor_name:  NOAA Global Drifter Program
        contributor_role:  Data Acquisition Center
        institution:       NOAA Atlantic Oceanographic and Meteorological Laboratory
        acknowledgement:   Lumpkin, Rick; Centurioni, Luca (2019). NOAA Global Dr...
        summary:           Global Drifter Program six-hourly data
        doi:               10.25921/7ntx-z961

    See Also
    --------
    :func:`gdp1h`
    """
    url = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/gdp_jul22_ragged_6h.nc#mode=bytes"
    return xr.open_dataset(url)


def mosaic() -> xr.Dataset:
    """Returns the MOSAiC sea-ice drift dataset as an Xarray dataset.

    The function will first look for the ragged-array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access.

    The upstream data is available at https://arcticdata.io/catalog/view/doi:10.18739/A2KP7TS83.

    Reference: Angela Bliss, Jennifer Hutchings, Philip Anderson, Philipp Anhaus,
    Hans Jakob Belter, Jørgen Berge, Vladimir Bessonov, Bin Cheng, Sylvia Cole,
    Dave Costa, Finlo Cottier, Christopher J Cox, Pedro R De La Torre, Dmitry V Divine,
    Gilbert Emzivat, Ying-Chih Fang, Steven Fons, Michael Gallagher, Maxime Geoffrey,
    Mats A Granskog, ... Guangyu Zuo. (2022). Sea ice drift tracks from the Distributed
    Network of autonomous buoys deployed during the Multidisciplinary drifting Observatory
    for the Study of Arctic Climate (MOSAiC) expedition 2019 - 2021. Arctic Data Center.
    doi:10.18739/A2KP7TS83.

    Returns
    -------
    xarray.Dataset
        MOSAiC sea-ice drift dataset as a ragged array

    Examples
    --------
    >>> from clouddrift.datasets import mosaic
    >>> ds = mosaic()
    >>> ds
    <xarray.Dataset>
    Dimensions:                     (obs: 1926226, traj: 216)
    Coordinates:
        time                        (obs) datetime64[ns] ...
        id                          (traj) object ...
    Dimensions without coordinates: obs, traj
    Data variables: (12/19)
        latitude                    (obs) float64 ...
        longitude                   (obs) float64 ...
        Deployment Leg              (traj) int64 ...
        DN Station ID               (traj) object ...
        IMEI                        (traj) object ...
        Deployment Date             (traj) datetime64[ns] ...
        ...                          ...
        Buoy Type                   (traj) object ...
        Manufacturer                (traj) object ...
        Model                       (traj) object ...
        PI                          (traj) object ...
        Data Authors                (traj) object ...
        rowsize                     (traj) int64 ...
    """
    clouddrift_path = (
        os.path.expanduser("~/.clouddrift")
        if not os.getenv("CLOUDDRIFT_PATH")
        else os.getenv("CLOUDDRIFT_PATH")
    )
    mosaic_path = f"{clouddrift_path}/data/mosaic.nc"
    if not os.path.exists(mosaic_path):
        print(f"{mosaic_path} not found; download from upstream repository.")
        ds = adapters.mosaic.to_xarray()
        os.makedirs(os.path.dirname(mosaic_path), exist_ok=True)
        ds.to_netcdf(mosaic_path)
    else:
        ds = xr.open_dataset(mosaic_path)
    return ds

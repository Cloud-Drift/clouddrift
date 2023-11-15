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


def glad() -> xr.Dataset:
    """Returns the Grand LAgrangian Deployment (GLAD) dataset as an Xarray dataset.

    The function will first look for the ragged-array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access.

    The upstream data is available at https://doi.org/10.7266/N7VD6WC8.

    Returns
    -------
    xarray.Dataset
        GLAD dataset as a ragged array

    Examples
    --------
    >>> from clouddrift.datasets import glad
    >>> ds = glad()
    >>> ds
    <xarray.Dataset>
    Dimensions:         (obs: 1602883, traj: 297)
    Coordinates:
      * time            (obs) datetime64[ns] 2012-07-20T01:15:00.143960 ... 2012-...
      * id              (traj) object 'CARTHE_001' 'CARTHE_002' ... 'CARTHE_451'
    Data variables:
      latitude        (obs) float32 ...
      longitude       (obs) float32 ...
      position_error  (obs) float32 ...
      u               (obs) float32 ...
      v               (obs) float32 ...
      velocity_error  (obs) float32 ...
      rowsize         (traj) int64 ...
    Attributes:
      title:        GLAD experiment CODE-style drifter trajectories (low-pass f...
      institution:  Consortium for Advanced Research on Transport of Hydrocarbo...
      source:       CODE-style drifters
      history:      Downloaded from https://data.gulfresearchinitiative.org/dat...
      references:   Özgökmen, Tamay. 2013. GLAD experiment CODE-style drifter t...

    Reference
    ---------
    Özgökmen, Tamay. 2013. GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records), northern Gulf of Mexico near DeSoto Canyon, July-October 2012. Distributed by: Gulf of Mexico Research Initiative Information and Data Cooperative (GRIIDC), Harte Research Institute, Texas A&M University–Corpus Christi. doi:10.7266/N7VD6WC8
    """
    clouddrift_path = (
        os.path.expanduser("~/.clouddrift")
        if not os.getenv("CLOUDDRIFT_PATH")
        else os.getenv("CLOUDDRIFT_PATH")
    )
    glad_path = f"{clouddrift_path}/data/glad.nc"
    if not os.path.exists(glad_path):
        print(f"{glad_path} not found; download from upstream repository.")
        ds = adapters.glad.to_xarray()
        os.makedirs(os.path.dirname(glad_path), exist_ok=True)
        ds.to_netcdf(glad_path)
    else:
        ds = xr.open_dataset(glad_path)
    return ds


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


def subsurface_floats() -> xr.Dataset:
    """Returns the subsurface floats dataset as an Xarray dataset.

    The function will first look for the ragged-array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access.

    The upstream data is available at
    https://www.aoml.noaa.gov/phod/float_traj/files/allFloats_12122017.mat.

    This dataset of subsurface float observations was compiled by the WOCE Subsurface
    Float Data Assembly Center (WFDAC) in Woods Hole maintained by Andree Ramsey and
    Heather Furey and copied to NOAA/AOML in October 2014 (version 1) and in December
    2017 (version 2). Subsequent updates will be included as additional appropriate
    float data, quality controlled by the appropriate principal investigators, is
    submitted for inclusion.

    Note that these observations are collected by ALACE/RAFOS/Eurofloat-style
    acoustically-tracked, neutrally-buoyant subsurface floats which collect data while
    drifting beneath the ocean surface. These data are the result of the effort and
    resources of many individuals and institutions. You are encouraged to acknowledge
    the work of the data originators and Data Centers in publications arising from use
    of these data.

    The float data were originally divided by project at the WFDAC. Here they have been
    compiled in a single Matlab data set. See here for more information on the variables
    contained in these files.

    Returns
    -------
    xarray.Dataset
        Subsurface floats dataset as a ragged array

    Examples
    --------
    >>> from clouddrift.datasets import subsurface_floats
    >>> ds = subsurface_floats()
    >>> ds
    <xarray.Dataset>
    Dimensions:   (traj: 2193, obs: 1402840)
    Coordinates:
        id        (traj) uint16 ...
        time      (obs) datetime64[ns] ...
    Dimensions without coordinates: traj, obs
    Data variables: (12/13)
        expList   (traj) object ...
        expName   (traj) object ...
        expOrg    (traj) object ...
        expPI     (traj) object ...
        indexExp  (traj) uint8 ...
        fltType   (traj) object ...
        ...        ...
        lon       (obs) float64 ...
        lat       (obs) float64 ...
        pres      (obs) float64 ...
        temp      (obs) float64 ...
        ve        (obs) float64 ...
        vn        (obs) float64 ...
    Attributes:
        title:            Subsurface float trajectories dataset
        history:          December 2017 (version 2)
        date_created:     2023-11-14T22:30:38.831656
        publisher_name:   WOCE Subsurface Float Data Assembly Center and NOAA AOML
        publisher_url:    https://www.aoml.noaa.gov/phod/float_traj/data.php
        licence:          freely available
        acknowledgement:  Maintained by Andree Ramsey and Heather Furey from the ...

    References
    ----------
    WOCE Subsurface Float Data Assembly Center (WFDAC) https://www.aoml.noaa.gov/phod/float_traj/index.php
    """

    clouddrift_path = (
        os.path.expanduser("~/.clouddrift")
        if not os.getenv("CLOUDDRIFT_PATH")
        else os.getenv("CLOUDDRIFT_PATH")
    )
    local_file = f"{clouddrift_path}/data/subsurface_floats.nc"
    if not os.path.exists(local_file):
        print(f"{local_file} not found; download from upstream repository.")
        ds = adapters.subsurface_floats.to_xarray()
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        ds.to_netcdf(local_file)
    else:
        ds = xr.open_dataset(local_file)
    return ds

"""
This module provides functions to easily access ragged array datasets. If the datasets are
not accessed via cloud storage platforms or are not found on the local filesystem,
they will be downloaded from their upstream repositories and stored for later access
(~/.clouddrift for UNIX-based systems).
"""

import os
from collections.abc import Callable

import xarray as xr

from clouddrift import adapters
from clouddrift.adapters.hurdat2 import _BasinOption


def gdp1h(decode_times: bool = True) -> xr.Dataset:
    """Returns the latest version of the NOAA Global Drifter Program (GDP) hourly
    dataset as a ragged array Xarray dataset.

    The data is accessed from zarr archive hosted on a public AWS S3 bucket accessible at
    https://registry.opendata.aws/noaa-oar-hourly-gdp/. Original data source from NOAA NCEI
    is https://doi.org/10.25921/x46c-3620).

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

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
        id                     (traj) int64 ...
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
    ds = xr.open_dataset(url, engine="zarr", decode_times=decode_times)
    ds = ds.rename_vars({"ID": "id"}).assign_coords({"id": ds.ID}).drop_vars(["ids"])
    return ds


def gdp6h(decode_times: bool = True) -> xr.Dataset:
    """Returns the NOAA Global Drifter Program (GDP) 6-hourly dataset as a ragged array
    Xarray dataset.

    The data is accessed from a public HTTPS server at NOAA's Atlantic
    Oceanographic and Meteorological Laboratory (AOML) accessible at
    https://www.aoml.noaa.gov/phod/gdp/index.php. It should be noted that the data loading
    method is platform dependent. Linux and Darwin (macOS) machines lazy load the datasets leveraging the
    byte-range feature of the netCDF-c library (dataset loading engine used by xarray).
    Windows machines download the entire dataset into a memory buffer which is then passed
    to xarray.

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

    Returns
    -------
    xarray.Dataset
        6-hourly GDP dataset as a ragged array

    Examples
    --------
    >>> from clouddrift.datasets import gdp6h
    >>> ds = gdp6h()
    >>> ds
    <xarray.Dataset> Size: 2GB
    Dimensions:                (traj: 27647, obs: 46535470)
    Coordinates:
    id                     (traj) int64 221kB ...
    time                   (obs) datetime64[ns] 372MB ...
    Dimensions without coordinates: traj, obs
    Data variables: (12/49)
    BuoyTypeManufacturer   (traj) |S20 553kB ...
    BuoyTypeSensorArray    (traj) |S20 553kB ...
    CurrentProgram         (traj) float64 221kB ...
    DeployingCountry       (traj) |S20 553kB ...
    DeployingShip          (traj) |S20 553kB ...
    DeploymentComments     (traj) |S20 553kB ...
    ...                     ...
    start_lon              (traj) float32 111kB ...
    temp                   (obs) float32 186MB ...
    typebuoy               (traj) |S10 276kB ...
    typedeath              (traj) int8 28kB ...
    ve                     (obs) float32 186MB ...
    vn                     (obs) float32 186MB ...
    Attributes: (12/18)
    Conventions:          CF-1.6
    acknowledgement:      Lumpkin, Rick; Centurioni, Luca (2019). NOAA Global...
    contributor_name:     NOAA Global Drifter Program
    contributor_role:     Data Acquisition Center
    date_created:         2024-04-04T13:44:01.176967
    doi:                  10.25921/7ntx-z961
    ...                   ...
    publisher_name:       GDP Drifter DAC
    publisher_url:        https://www.aoml.noaa.gov/phod/gdp
    summary:              Global Drifter Program six-hourly data
    time_coverage_end:    2023-10-18:18:00:00Z
    time_coverage_start:  1979-02-15:00:00:00Z
    title:                Global Drifter Program drifting buoy collection

    See Also
    --------
    :func:`gdp1h`
    """
    url = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/experimental/gdp6h_ragged_sep23.zarr"
    ds = xr.open_dataset(url, decode_times=decode_times, engine="zarr")
    return ds


def glad(decode_times: bool = True) -> xr.Dataset:
    """Returns the Grand LAgrangian Deployment (GLAD) dataset as a ragged array
    Xarray dataset.

    The function will first look for the ragged-array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access.

    The upstream data is available at https://doi.org/10.7266/N7VD6WC8.

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

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
      time            (obs) datetime64[ns] ...
      id              (traj) object ...
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
    return _dataset_filecache("glad.nc", decode_times, adapters.glad.to_xarray)


def hurdat2(basin: _BasinOption = "both", decode_times: bool = True) -> xr.DataArray:
    """Returns the revised hurricane database (HURDAT2) as a ragged array xarray dataset.

    The function will first look for the ragged array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access.

    The upstream data is available at https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html.

    Parameters
    ----------
    basin : "atlantic", "pacific", "both" (default)
        Specify which ocean basin to download storm data for. Current available options are the
        Atlantic Ocean "atlantic", Pacific Ocean "pacific" and, both "both" to download
        storm data for both oceans.
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

    Returns
    -------
    xarray.Dataset
        HURDAT2 dataset as a ragged array.

    Standard usage of the dataset.

    >>> from clouddrift.datasets import hurdat2
    >>> ds = hurdat2()
    >>> ds
    <xarray.Dataset>
    Dimensions:                          (traj: ..., obs: ...)
    Coordinates:
        atcf_identifier                  (traj) <U8 ...
        time                             (obs) datetime64[ns] ...
    Dimensions without coordinates: traj, obs
    Data variables: (12/23)
        basin                            (traj) <U2 ...
        year                             (traj) int64 ...
        rowsize                          (traj) int64 ...
        record_identifier                (obs) <U1 ...
        system_status                    (obs) <U2 ...
        ...                               ...
        max_med_wind_radius_nw           (obs) float64 ...
        max_high_wind_radius_ne          (obs) float64 ...
        max_high_wind_radius_se          (obs) float64 ...
        max_high_wind_radius_sw          (obs) float64 ...
        max_high_wind_radius_nw          (obs) float64 ...
        max_sustained_wind_speed_radius  (obs) float64 ...
    Attributes:
        title:            HURricane DATa 2nd generation (HURDAT2)
        date_created:     ...
        publisher_name:   NOAA AOML Hurricane Research Division
        publisher_email:  AOML.HRDWebmaster@noaa.gov
        publisher_url:    https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html
        institution:      NOAA Atlantic Oceanographic and Meteorological Laboratory
        summary:          The National Hurricane Center (NHC) conducts a post-sto...
    ...

    To only retrieve records for the Atlantic Ocean basin.

    >>> from clouddrift.datasets import hurdat2
    >>> ds = hurdat2(basin="atlantic")
    >>> ds
    <xarray.Dataset>
    ...

    Reference
    ---------
    https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html.
    """
    return _dataset_filecache(
        f"hurdat2_{basin}.nc",
        decode_times,
        lambda: adapters.hurdat2.to_raggedarray(basin).to_xarray(),
    )


def mosaic(decode_times: bool = True) -> xr.Dataset:
    """Returns the MOSAiC sea-ice drift dataset as a ragged array Xarray dataset.

    The function will first look for the ragged-array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access.

    The upstream data is available at https://arcticdata.io/catalog/view/doi:10.18739/A2KP7TS83.

    Reference
    ---------
    Angela Bliss, Jennifer Hutchings, Philip Anderson, Philipp Anhaus,
    Hans Jakob Belter, Jørgen Berge, Vladimir Bessonov, Bin Cheng, Sylvia Cole,
    Dave Costa, Finlo Cottier, Christopher J Cox, Pedro R De La Torre, Dmitry V Divine,
    Gilbert Emzivat, Ying-Chih Fang, Steven Fons, Michael Gallagher, Maxime Geoffrey,
    Mats A Granskog, ... Guangyu Zuo. (2022). Sea ice drift tracks from the Distributed
    Network of autonomous buoys deployed during the Multidisciplinary drifting Observatory
    for the Study of Arctic Climate (MOSAiC) expedition 2019 - 2021. Arctic Data Center.
    doi:10.18739/A2KP7TS83.

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

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
    return _dataset_filecache("mosaic.nc", decode_times, adapters.mosaic.to_xarray)


def spotters(decode_times: bool = True) -> xr.Dataset:
    """Returns the Sofar Ocean Spotter drifters ragged array dataset as an Xarray dataset.

    The data is accessed from a zarr archive hosted on a public AWS S3 bucket accessible
    at https://sofar-spotter-archive.s3.amazonaws.com/spotter_data_bulk_zarr.

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

    Returns
    -------
    xarray.Dataset
        Sofar ocean floats dataset as a ragged array

    Examples
    --------
    >>> from clouddrift.datasets import spotters
    >>> ds = spotters()
    >>> ds
    <xarray.Dataset>
    Dimensions:                (index: 6390651, trajectory: 871)
    Coordinates:
        time                   (index) datetime64[ns] ...
      * trajectory             (trajectory) object 'SPOT-010001' ... 'SPOT-1975'
    Dimensions without coordinates: index
    Data variables:
        latitude               (index) float64 ...
        longitude              (index) float64 ...
        meanDirection          (index) float64 ...
        meanDirectionalSpread  (index) float64 ...
        meanPeriod             (index) float64 ...
        peakDirection          (index) float64 ...
        peakDirectionalSpread  (index) float64 ...
        peakPeriod             (index) float64 ...
        rowsize                (trajectory) int64 ...
        significantWaveHeight  (index) float64 ...
    Attributes:
        author:         Isabel A. Houghton
        creation_date:  2023-10-18 00:43:55.333537
        email:          isabel.houghton@sofarocean.com
        institution:    Sofar Ocean
        references:     https://content.sofarocean.com/hubfs/Spotter%20product%20...
        source:         Spotter wave buoy
        title:          Sofar Spotter Data Archive - Bulk Wave Parameters
    """
    url = "https://sofar-spotter-archive.s3.amazonaws.com/spotter_data_bulk_zarr"
    return xr.open_dataset(url, engine="zarr", decode_times=decode_times)


def subsurface_floats(decode_times: bool = True) -> xr.Dataset:
    """Returns the subsurface floats dataset as a ragged array Xarray dataset.

    The data is accessed from a public HTTPS server at NOAA's Atlantic
    Oceanographic and Meteorological Laboratory (AOML) accessible at
    https://www.aoml.noaa.gov/phod/gdp/index.php.

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

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

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
        license:          freely available
        acknowledgement:  Maintained by Andree Ramsey and Heather Furey from the ...

    References
    ----------
    WOCE Subsurface Float Data Assembly Center (WFDAC) https://www.aoml.noaa.gov/phod/float_traj/index.php
    """
    return _dataset_filecache(
        "subsurface_floats.nc", decode_times, adapters.subsurface_floats.to_xarray
    )


def yomaha(decode_times: bool = True) -> xr.Dataset:
    """Returns the YoMaHa dataset as a ragged array Xarray dataset.

    The function will first look for the ragged-array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access. The upstream
    data is available at http://apdrc.soest.hawaii.edu/projects/yomaha/.

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

    Returns
    -------
    xarray.Dataset
        YoMaHa'07 dataset as a ragged array

    Examples
    --------

    >>> from clouddrift.datasets import yomaha
    >>> ds = yomaha()
    >>> ds
    <xarray.Dataset>
    Dimensions:     (obs: 1926743, traj: 12196)
    Coordinates:
        time_d      (obs) datetime64[ns] ...
        time_s      (obs) datetime64[ns] ...
        time_lp     (obs) datetime64[ns] ...
        time_lc     (obs) datetime64[ns] ...
        id          (traj) int64 ...
    Dimensions without coordinates: obs, traj
    Data variables: (12/27)
        lon_d       (obs) float64 ...
        lat_d       (obs) float64 ...
        pres_d      (obs) float32 ...
        ve_d        (obs) float32 ...
        vn_d        (obs) float32 ...
        err_ve_d    (obs) float32 ...
        ...          ...
        cycle       (obs) int64 ...
        time_inv    (obs) int64 ...
        rowsize     (traj) int64 ...
        wmo_id      (traj) int64 ...
        dac_id      (traj) int64 ...
        float_type  (traj) int64 ...
    Attributes:
        title:           YoMaHa'07: Velocity data assessed from trajectories of A...
        history:         Dataset updated on Tue Jun 28 03:14:34 HST 2022
        date_created:    2023-12-08T00:52:08.478075
        publisher_name:  Asia-Pacific Data Research Center
        publisher_url:   http://apdrc.soest.hawaii.edu/index.php
        license:         Creative Commons Attribution 4.0 International License..

    Reference
    ---------
    Lebedev, K. V., Yoshinari, H., Maximenko, N. A., & Hacker, P. W. (2007). Velocity data
    assessed  from trajectories of Argo floats at parking level and at the sea
    surface. IPRC Technical Note, 4(2), 1-16.
    """
    return _dataset_filecache("yomaha.nc", decode_times, adapters.yomaha.to_xarray)


def andro(decode_times: bool = False) -> xr.Dataset:
    """Returns the ANDRO as a ragged array Xarray dataset.

    The function will first look for the ragged-array dataset on the local
    filesystem. If it is not found, the dataset will be downloaded using the
    corresponding adapter function and stored for later access. The upstream
    data is available at https://www.seanoe.org/data/00360/47077/.

    Parameters
    ----------
    decode_times : bool, optional
        If True, decode the time coordinate into a datetime object. If False, the time
        coordinate will be an int64 or float64 array of increments since the origin
        time indicated in the units attribute. Default is True.

    Returns
    -------
    xarray.Dataset
        ANDRO dataset as a ragged array
    Examples
    --------
    >>> from clouddrift.datasets import andro
    >>> ds = andro()
    >>> ds
    <xarray.Dataset>
    Dimensions:     (obs: 1360753, traj: 9996)
    Coordinates:
        time_d      (obs) datetime64[ns] ...
        time_s      (obs) datetime64[ns] ...
        time_lp     (obs) datetime64[ns] ...
        time_lc     (obs) datetime64[ns] ...
        id          (traj) int64 ...
    Dimensions without coordinates: obs, traj
    Data variables: (12/33)
        lon_d       (obs) float64 ...
        lat_d       (obs) float64 ...
        pres_d      (obs) float32 ...
        temp_d      (obs) float32 ...
        sal_d       (obs) float32 ...
        ve_d        (obs) float32 ...
        ...          ...
        lon_lc      (obs) float64 ...
        lat_lc      (obs) float64 ...
        surf_fix    (obs) int64 ...
        cycle       (obs) int64 ...
        profile_id  (obs) float32 ...
        rowsize     (traj) int64 ...
    Attributes:
        title:           ANDRO: An Argo-based deep displacement dataset
        history:         2022-03-04
        date_created:    2023-12-08T00:52:00.937120
        publisher_name:  SEANOE (SEA scieNtific Open data Edition)
        publisher_url:   https://www.seanoe.org/data/00360/47077/
        license:         freely available

    Reference
    ---------
    Ollitrault Michel, Rannou Philippe, Brion Emilie, Cabanes Cecile, Piron Anne, Reverdin Gilles,
    Kolodziejczyk Nicolas (2022). ANDRO: An Argo-based deep displacement dataset.
    SEANOE. https://doi.org/10.17882/47077
    """
    return _dataset_filecache("andro.nc", decode_times, adapters.andro.to_xarray)


def _dataset_filecache(
    filename: str, decode_times: bool, get_ds: Callable[[], xr.Dataset]
):
    clouddrift_path = (
        os.path.expanduser("~/.clouddrift")
        if not os.getenv("CLOUDDRIFT_PATH")
        else os.getenv("CLOUDDRIFT_PATH")
    )
    fp = f"{clouddrift_path}/data/{filename}"
    if not os.path.exists(fp):
        print(f"{fp} not found; download from upstream repository.")
        ds = get_ds()
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        ds.to_netcdf(fp)
    return xr.open_dataset(fp, decode_times=decode_times)

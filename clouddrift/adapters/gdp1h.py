"""
This module provides functions and metadata that can be used to convert the
hourly Global Drifter Program (GDP) data to a ``clouddrift.RaggedArray``
instance.
"""

import logging
import os
import re
import tempfile
import urllib.request
import warnings
from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr

import clouddrift.adapters.gdp as gdp
from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

GDP_VERSION = "2.01"


GDP_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/hourly_product/v2.01"
GDP_DATA_URL_EXPERIMENTAL = (
    "https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/experimental"
)


GDP_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdp")
GDP_TMP_PATH_EXPERIMENTAL = os.path.join(tempfile.gettempdir(), "clouddrift", "gdp_exp")
GDP_DATA = [
    "lon",
    "lat",
    "ve",
    "vn",
    "err_lat",
    "err_lon",
    "err_ve",
    "err_vn",
    "gap",
    "sst",
    "sst1",
    "sst2",
    "err_sst",
    "err_sst1",
    "err_sst2",
    "flg_sst",
    "flg_sst1",
    "flg_sst2",
    "drogue_status",
]

_logger = logging.getLogger(__name__)


def download(
    url: str,
    tmp_path: str,
    drifter_ids: Union[list[int], None] = None,
    n_random_id: Union[int, None] = None,
):
    """Download individual NetCDF files from the AOML server.

    Parameters
    ----------
    url : str
        URL from which to download the data.
    tmp_path : str
        Path to the directory where the individual NetCDF files are stored.

    drifter_ids : list, optional
        List of drifter to retrieve (Default: all)
    n_random_id : int, optional
        Randomly select n_random_id drifter IDs to download (Default: None)
    Returns
    -------
    out : list
        List of retrieved drifters
    """
    _logger.debug(f"Downloading GDP hourly data from ({url}) to ({tmp_path})")

    # Create a temporary directory if doesn't already exists.
    os.makedirs(tmp_path, exist_ok=True)
    pattern = "drifter_hourly_[0-9]*.nc"
    filename_pattern = "drifter_hourly_{id}.nc"

    # retrieve all drifter ID numbers
    if drifter_ids is None:
        urlpath = urllib.request.urlopen(url)
        string = urlpath.read().decode("utf-8")
        filelist: Sequence[str] = re.compile(pattern).findall(string)  # noqa: F821
    else:
        filelist = [filename_pattern.format(id=did) for did in drifter_ids]
    filelist = list(np.unique(filelist))

    # retrieve only a subset of n_random_id trajectories
    if n_random_id:
        if n_random_id > len(filelist):
            _logger.warn(
                f"Retrieving all listed trajectories because {n_random_id} is larger than the {len(filelist)} listed trajectories."
            )
        else:
            rng = np.random.RandomState(42)
            filelist = sorted(rng.choice(filelist, n_random_id, replace=False))

    download_with_progress(
        [(f"{url}/{f}", os.path.join(tmp_path, f), None) for f in filelist]
    )
    # Download the metadata so we can order the drifter IDs by end date.
    gdp_metadata = gdp.get_gdp_metadata()

    return gdp.order_by_date(
        gdp_metadata, [int(f.split("_")[-1].removesuffix(".nc")) for f in filelist]
    )


def preprocess(index: int, **kwargs) -> xr.Dataset:
    """Extract and preprocess the Lagrangian data and attributes.

    This function takes an identification number that can be used to create a
    file or url pattern or select data from a Dataframe. It then preprocesses
    the data and returns a clean Xarray Dataset.

    Parameters
    ----------
    index : int
        Drifter's identification number

    Returns
    -------
    ds : xr.Dataset
        Xarray Dataset containing the data and attributes
    """
    fp = os.path.join(kwargs["tmp_path"], kwargs["filename_pattern"].format(id=index))
    ds = xr.open_dataset(
        fp,
        decode_times=False,
        decode_coords=False,
    )

    # parse the date with custom function
    ds["deploy_date"].data = gdp.decode_date(np.array([ds.deploy_date.data[0]]))
    ds["end_date"].data = gdp.decode_date(np.array([ds.end_date.data[0]]))
    ds["drogue_lost_date"].data = gdp.decode_date(
        np.array([ds.drogue_lost_date.data[0]])
    )
    ds["time"].data = gdp.decode_date(np.array([ds.time.data[0]]))

    # convert fill values to nan
    for var in [
        "err_lon",
        "err_lat",
        "err_ve",
        "err_vn",
        "sst",
        "sst1",
        "sst2",
        "err_sst",
        "err_sst1",
        "err_sst2",
    ]:
        try:
            ds[var].data = gdp.fill_values(ds[var].data)
        except KeyError:
            warnings.warn(f"Variable {var} not found; skipping.")

    # fix missing values stored as str
    for var in [
        "longitude",
        "latitude",
        "err_lat",
        "err_lon",
        "ve",
        "vn",
        "err_ve",
        "err_vn",
        "sst",
        "sst1",
        "sst2",
    ]:
        try:
            ds[var].encoding["missing value"] = -1e-34
        except KeyError:
            warnings.warn(f"Variable {var} not found in upstream data; skipping.")

    # convert type of some variable
    target_dtype = {
        "ID": "int64",
        "WMO": "int32",
        "expno": "int32",
        "typedeath": "int8",
        "flg_sst": "int8",
        "flg_sst1": "int8",
        "flg_sst2": "int8",
    }

    for var in target_dtype.keys():
        if var in ds.keys():
            ds[var].data = ds[var].data.astype(target_dtype[var])
        else:
            warnings.warn(f"Variable {var} not found in upstream data; skipping.")

    # new variables
    ds["drogue_status"] = (
        ["traj", "obs"],
        [gdp.drogue_presence(ds.drogue_lost_date.data, ds.time.data[0])],
    )

    # convert attributes to variable
    ds["location_type"] = (
        ("traj"),
        [False if ds.get("location_type") == "Argos" else True],
    )  # 0 for Argos, 1 for GPS
    ds["DeployingShip"] = (("traj"), gdp.cut_str(ds.DeployingShip, 20))
    ds["DeploymentStatus"] = (
        ("traj"),
        gdp.cut_str(ds.DeploymentStatus, 20),
    )
    ds["BuoyTypeManufacturer"] = (
        ("traj"),
        gdp.cut_str(ds.BuoyTypeManufacturer, 20),
    )
    ds["BuoyTypeSensorArray"] = (
        ("traj"),
        gdp.cut_str(ds.BuoyTypeSensorArray, 20),
    )
    ds["CurrentProgram"] = (
        ("traj"),
        np.array([gdp.str_to_float(ds.CurrentProgram, -1)], dtype=np.int32),
    )
    ds["PurchaserFunding"] = (
        ("traj"),
        gdp.cut_str(ds.PurchaserFunding, 20),
    )
    ds["SensorUpgrade"] = (("traj"), gdp.cut_str(ds.SensorUpgrade, 20))
    ds["Transmissions"] = (("traj"), gdp.cut_str(ds.Transmissions, 20))
    ds["DeployingCountry"] = (
        ("traj"),
        gdp.cut_str(ds.DeployingCountry, 20),
    )
    ds["DeploymentComments"] = (
        ("traj"),
        np.array([gdp.str_to_float(ds.ManufactureYear, -1)], dtype=np.int16),
    )
    ds["ManufactureMonth"] = (
        ("traj"),
        np.array([gdp.str_to_float(ds.ManufactureMonth, -1)], dtype=np.int16),
    )
    ds["ManufactureSensorType"] = (
        ("traj"),
        gdp.cut_str(ds.ManufactureSensorType, 20),
    )
    ds["ManufactureVoltage"] = (
        ("traj"),
        np.array([gdp.str_to_float(ds.ManufactureVoltage[:-2], -1)], dtype=np.int16),
    )  # e.g. 56 V
    ds["FloatDiameter"] = (
        ("traj"),
        [gdp.str_to_float(ds.FloatDiameter[:-3])],
    )  # e.g. 35.5 cm
    ds["SubsfcFloatPresence"] = (
        ("traj"),
        np.array([gdp.str_to_float(ds.SubsfcFloatPresence)], dtype="bool"),
    )
    ds["DrogueType"] = (("traj"), gdp.cut_str(ds.DrogueType, 7))
    ds["DrogueLength"] = (
        ("traj"),
        [gdp.str_to_float(ds.DrogueLength[:-2])],
    )  # e.g. 4.8 m
    ds["DrogueBallast"] = (
        ("traj"),
        [gdp.str_to_float(ds.DrogueBallast[:-3])],
    )  # e.g. 1.4 kg
    ds["DragAreaAboveDrogue"] = (
        ("traj"),
        [gdp.str_to_float(ds.DragAreaAboveDrogue[:-4])],
    )  # 10.66 m^2
    ds["DragAreaOfDrogue"] = (
        ("traj"),
        [gdp.str_to_float(ds.DragAreaOfDrogue[:-4])],
    )  # e.g. 416.6 m^2
    ds["DragAreaRatio"] = (
        ("traj"),
        [gdp.str_to_float(ds.DragAreaRatio)],
    )  # e.g. 39.08
    ds["DrogueCenterDepth"] = (
        ("traj"),
        [gdp.str_to_float(ds.DrogueCenterDepth[:-2])],
    )  # e.g. 20.0 m
    ds["DrogueDetectSensor"] = (
        ("traj"),
        gdp.cut_str(ds.DrogueDetectSensor, 20),
    )

    # vars attributes
    vars_attrs = {
        "ID": {"long_name": "Global Drifter Program Buoy ID", "units": "-"},
        "longitude": {"long_name": "Longitude", "units": "degrees_east"},
        "latitude": {"long_name": "Latitude", "units": "degrees_north"},
        "time": {"long_name": "Time", "units": "seconds since 1970-01-01 00:00:00"},
        "rowsize": {
            "long_name": "Number of observations per trajectory",
            "sample_dimension": "obs",
            "units": "-",
        },
        "location_type": {
            "long_name": "Satellite-based location system",
            "units": "-",
            "comments": "0 (Argos), 1 (GPS)",
        },
        "WMO": {
            "long_name": "World Meteorological Organization buoy identification number",
            "units": "-",
        },
        "expno": {"long_name": "Experiment number", "units": "-"},
        "deploy_date": {
            "long_name": "Deployment date and time",
            "units": "seconds since 1970-01-01 00:00:00",
        },
        "deploy_lon": {"long_name": "Deployment longitude", "units": "degrees_east"},
        "deploy_lat": {"long_name": "Deployment latitude", "units": "degrees_north"},
        "start_date": {
            "long_name": "First good date and time derived by DAC quality control",
            "units": "seconds since 1970-01-01 00:00:00",
        },
        "start_lon": {
            "long_name": "First good longitude derived by DAC quality control",
            "units": "degrees_east",
        },
        "start_lat": {
            "long_name": "Last good latitude derived by DAC quality control",
            "units": "degrees_north",
        },
        "end_date": {
            "long_name": "Last good date and time derived by DAC quality control",
            "units": "seconds since 1970-01-01 00:00:00",
        },
        "end_lon": {
            "long_name": "Last good longitude derived by DAC quality control",
            "units": "degrees_east",
        },
        "end_lat": {
            "long_name": "Last good latitude derived by DAC quality control",
            "units": "degrees_north",
        },
        "drogue_lost_date": {
            "long_name": "Date and time of drogue loss",
            "units": "seconds since 1970-01-01 00:00:00",
        },
        "typedeath": {
            "long_name": "Type of death",
            "units": "-",
            "comments": "0 (buoy still alive), 1 (buoy ran aground), 2 (picked up by vessel), 3 (stop transmitting), 4 (sporadic transmissions), 5 (bad batteries), 6 (inactive status)",
        },
        "typebuoy": {
            "long_name": "Buoy type (see https://www.aoml.noaa.gov/phod/dac/dirall.html)",
            "units": "-",
        },
        "DeployingShip": {"long_name": "Name of deployment ship", "units": "-"},
        "DeploymentStatus": {"long_name": "Deployment status", "units": "-"},
        "BuoyTypeManufacturer": {"long_name": "Buoy type manufacturer", "units": "-"},
        "BuoyTypeSensorArray": {"long_name": "Buoy type sensor array", "units": "-"},
        "CurrentProgram": {
            "long_name": "Current Program",
            "units": "-",
            "_FillValue": "-1",
        },
        "PurchaserFunding": {"long_name": "Purchaser funding", "units": "-"},
        "SensorUpgrade": {"long_name": "Sensor upgrade", "units": "-"},
        "Transmissions": {"long_name": "Transmissions", "units": "-"},
        "DeployingCountry": {"long_name": "Deploying country", "units": "-"},
        "DeploymentComments": {"long_name": "Deployment comments", "units": "-"},
        "ManufactureYear": {
            "long_name": "Manufacture year",
            "units": "-",
            "_FillValue": "-1",
        },
        "ManufactureMonth": {
            "long_name": "Manufacture month",
            "units": "-",
            "_FillValue": "-1",
        },
        "ManufactureSensorType": {"long_name": "Manufacture Sensor Type", "units": "-"},
        "ManufactureVoltage": {
            "long_name": "Manufacture voltage",
            "units": "V",
            "_FillValue": "-1",
        },
        "FloatDiameter": {"long_name": "Diameter of surface floater", "units": "cm"},
        "SubsfcFloatPresence": {"long_name": "Subsurface Float Presence", "units": "-"},
        "DrogueType": {"drogue_type": "Drogue Type", "units": "-"},
        "DrogueLength": {"long_name": "Length of drogue.", "units": "m"},
        "DrogueBallast": {
            "long_name": "Weight of the drogue's ballast.",
            "units": "kg",
        },
        "DragAreaAboveDrogue": {"long_name": "Drag area above drogue.", "units": "m^2"},
        "DragAreaOfDrogue": {"long_name": "Drag area drogue.", "units": "m^2"},
        "DragAreaRatio": {"long_name": "Drag area ratio", "units": "m"},
        "DrogueCenterDepth": {
            "long_name": "Average depth of the drogue.",
            "units": "m",
        },
        "DrogueDetectSensor": {"long_name": "Drogue detection sensor", "units": "-"},
        "ve": {"long_name": "Eastward velocity", "units": "m/s"},
        "vn": {"long_name": "Northward velocity", "units": "m/s"},
        "gap": {
            "long_name": "Time interval between previous and next location",
            "units": "s",
        },
        "err_lat": {
            "long_name": "95% confidence interval in latitude",
            "units": "degrees_north",
        },
        "err_lon": {
            "long_name": "95% confidence interval in longitude",
            "units": "degrees_east",
        },
        "err_ve": {
            "long_name": "95% confidence interval in eastward velocity",
            "units": "m/s",
        },
        "err_vn": {
            "long_name": "95% confidence interval in northward velocity",
            "units": "m/s",
        },
        "drogue_status": {
            "long_name": "Status indicating the presence of the drogue",
            "units": "-",
            "flag_values": "1,0",
            "flag_meanings": "drogued, undrogued",
        },
        "sst": {
            "long_name": "Fitted sea water temperature",
            "units": "Kelvin",
            "comments": "Estimated near-surface sea water temperature from drifting buoy measurements. It is the sum of the fitted near-surface non-diurnal sea water temperature and fitted diurnal sea water temperature anomaly. Discrepancies may occur because of rounding.",
        },
        "sst1": {
            "long_name": "Fitted non-diurnal sea water temperature",
            "units": "Kelvin",
            "comments": "Estimated near-surface non-diurnal sea water temperature from drifting buoy measurements",
        },
        "sst2": {
            "long_name": "Fitted diurnal sea water temperature anomaly",
            "units": "Kelvin",
            "comments": "Estimated near-surface diurnal sea water temperature anomaly from drifting buoy measurements",
        },
        "err_sst": {
            "long_name": "Standard uncertainty of fitted sea water temperature",
            "units": "Kelvin",
            "comments": "Estimated one standard error of near-surface sea water temperature estimate from drifting buoy measurements",
        },
        "err_sst1": {
            "long_name": "Standard uncertainty of fitted non-diurnal sea water temperature",
            "units": "Kelvin",
            "comments": "Estimated one standard error of near-surface non-diurnal sea water temperature estimate from drifting buoy measurements",
        },
        "err_sst2": {
            "long_name": "Standard uncertainty of fitted diurnal sea water temperature anomaly",
            "units": "Kelvin",
            "comments": "Estimated one standard error of near-surface diurnal sea water temperature anomaly estimate from drifting buoy measurements",
        },
        "flg_sst": {
            "long_name": "Fitted sea water temperature quality flag",
            "units": "-",
            "flag_values": "0, 1, 2, 3, 4, 5",
            "flag_meanings": "no-estimate, no-uncertainty-estimate, estimate-not-in-range-uncertainty-not-in-range, estimate-not-in-range-uncertainty-in-range estimate-in-range-uncertainty-not-in-range, estimate-in-range-uncertainty-in-range",
        },
        "flg_sst1": {
            "long_name": "Fitted non-diurnal sea water temperature quality flag",
            "units": "-",
            "flag_values": "0, 1, 2, 3, 4, 5",
            "flag_meanings": "no-estimate, no-uncertainty-estimate, estimate-not-in-range-uncertainty-not-in-range, estimate-not-in-range-uncertainty-in-range estimate-in-range-uncertainty-not-in-range, estimate-in-range-uncertainty-in-range",
        },
        "flg_sst2": {
            "long_name": "Fitted diurnal sea water temperature anomaly quality flag",
            "units": "-",
            "flag_values": "0, 1, 2, 3, 4, 5",
            "flag_meanings": "no-estimate, no-uncertainty-estimate, estimate-not-in-range-uncertainty-not-in-range, estimate-not-in-range-uncertainty-in-range estimate-in-range-uncertainty-not-in-range, estimate-in-range-uncertainty-in-range",
        },
    }

    # global attributes
    attrs = {
        "title": "Global Drifter Program hourly drifting buoy collection",
        "history": f"version {GDP_VERSION}. Metadata from dirall.dat and deplog.dat",
        "Conventions": "CF-1.6",
        "time_coverage_start": "",
        "time_coverage_end": "",
        "date_created": datetime.now().isoformat(),
        "publisher_name": "GDP Drifter DAC",
        "publisher_email": "aoml.dftr@noaa.gov",
        "publisher_url": "https://www.aoml.noaa.gov/phod/gdp",
        "license": "freely available",
        "processing_level": "Level 2 QC by GDP drifter DAC",
        "metadata_link": "https://www.aoml.noaa.gov/phod/dac/dirall.html",
        "contributor_name": "NOAA Global Drifter Program",
        "contributor_role": "Data Acquisition Center",
        "institution": "NOAA Atlantic Oceanographic and Meteorological Laboratory",
        "acknowledgement": "Elipot, Shane; Sykulski, Adam; Lumpkin, Rick; Centurioni, Luca; Pazos, Mayra (2022). Hourly location, current velocity, and temperature collected from Global Drifter Program drifters world-wide. [indicate subset used]. NOAA National Centers for Environmental Information. Dataset. https://doi.org/10.25921/x46c-3620. Accessed [date]. Elipot et al. (2022): A Dataset of Hourly Sea Surface Temperature From Drifting Buoys, Scientific Data, 9, 567, https://dx.doi.org/10.1038/s41597-022-01670-2. Elipot et al. (2016): A global surface drifter dataset at hourly resolution, J. Geophys. Res.-Oceans, 121, https://dx.doi.org/10.1002/2016JC011716.",
        "summary": "Global Drifter Program hourly data",
        "doi": "10.25921/x46c-3620",
    }

    # set attributes
    for var in vars_attrs.keys():
        if var in ds.keys():
            ds[var].attrs = vars_attrs[var]
        else:
            warnings.warn(f"Variable {var} not found in upstream data; skipping.")
    ds.attrs = attrs

    # rename variables
    ds = ds.rename_vars({"longitude": "lon", "latitude": "lat", "ID": "id"})

    # Cast float64 variables to float32 to reduce memory footprint.
    ds = gdp.cast_float64_variables_to_float32(ds)

    return ds


def to_raggedarray(
    drifter_ids: Optional[list[int]] = None,
    n_random_id: Optional[int] = None,
    url: str = GDP_DATA_URL,
    tmp_path: Optional[str] = None,
) -> RaggedArray:
    """Download and process individual GDP hourly files and return a RaggedArray
    instance with the data.

    Parameters
    ----------
    drifter_ids : list[int], optional
        List of drifters to retrieve (Default: all)
    n_random_id : list[int], optional
        Randomly select n_random_id drifter NetCDF files
    url : str
        URL from which to download the data (Default: GDP_DATA_URL).
        Alternatively, it can be GDP_DATA_URL_EXPERIMENTAL.
    tmp_path : str, optional
        Path to the directory where the individual NetCDF files are stored
        (default varies depending on operating system; /tmp/clouddrift/gdp on Linux)

    Returns
    -------
    out : RaggedArray
        A RaggedArray instance of the requested dataset

    Examples
    --------

    Invoke `to_raggedarray` without any arguments to download all drifter data
    from the 2.01 GDP feed:

    >>> from clouddrift.adapters.gdp1h import to_raggedarray
    >>> ra = to_raggedarray()

    To download a random sample of 100 drifters, for example for development
    or testing, use the `n_random_id` argument:

    >>> ra = to_raggedarray(n_random_id=100)

    To download a specific list of drifters, use the `drifter_ids` argument:

    >>> ra = to_raggedarray(drifter_ids=[44136, 54680, 83463])

    To download the experimental 2.01 GDP feed, use the `url` argument to
    specify the experimental feed URL:

    >>> from clouddrift.adapters.gdp1h import GDP_DATA_URL_EXPERIMENTAL, to_raggedarray
    >>> ra = to_raggedarray(url=GDP_DATA_URL_EXPERIMENTAL)

    Finally, `to_raggedarray` returns a `RaggedArray` instance which provides
    a convenience method to emit a `xarray.Dataset` instance:

    >>> ds = ra.to_xarray()

    To write the ragged array dataset to a NetCDF file on disk, do

    >>> ds.to_netcdf("gdp1h.nc", format="NETCDF4")

    Alternatively, to write the ragged array to a Parquet file, first create
    it as an Awkward Array:

    >>> arr = ra.to_awkward()
    >>> arr.to_parquet("gdp1h.parquet")
    """

    # adjust the tmp_path if using the experimental source
    if tmp_path is None:
        tmp_path = GDP_TMP_PATH if url == GDP_DATA_URL else GDP_TMP_PATH_EXPERIMENTAL

    ids = download(url, tmp_path, drifter_ids, n_random_id)
    filename_pattern = "drifter_hourly_{id}.nc"

    ra = RaggedArray.from_files(
        indices=ids,
        preprocess_func=preprocess,
        name_coords=gdp.GDP_COORDS,
        name_meta=gdp.GDP_METADATA,
        name_data=GDP_DATA,
        name_dims=gdp.GDP_DIMS,
        rowsize_func=gdp.rowsize,
        filename_pattern=filename_pattern,
        tmp_path=tmp_path,
    )

    # set dynamic global attributes
    if ra.attrs_global:
        ra.attrs_global["time_coverage_start"] = (
            f"{datetime(1970,1,1) + timedelta(seconds=int(np.min(ra.coords['time']))):%Y-%m-%d:%H:%M:%SZ}"
        )
        ra.attrs_global["time_coverage_end"] = (
            f"{datetime(1970,1,1) + timedelta(seconds=int(np.max(ra.coords['time']))):%Y-%m-%d:%H:%M:%SZ}"
        )

    return ra

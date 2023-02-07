"""
This module provides functions and metadata that can be used to convert the
hourly Global Drifter Program (GDP) data to a ``clouddrift.RaggedArray`` instance.
"""

from ..dataformat import RaggedArray
import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr
import urllib.request
import concurrent.futures
import re
import tempfile
from tqdm import tqdm
from typing import Optional
import os
import warnings

GDP_VERSION = "2.00"
GDP_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/v2.00/netcdf/"
GDP_FILENAME_PATTERN = "drifter_{id}.nc"
GDP_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdp")

GDP_COORDS = [
    "ids",
    "time",
]
GDP_METADATA = [
    "ID",
    "rowsize",
    "WMO",
    "expno",
    "deploy_date",
    "deploy_lat",
    "deploy_lon",
    "end_date",
    "end_lat",
    "end_lon",
    "drogue_lost_date",
    "typedeath",
    "typebuoy",
    "location_type",
    "DeployingShip",
    "DeploymentStatus",
    "BuoyTypeManufacturer",
    "BuoyTypeSensorArray",
    "CurrentProgram",
    "PurchaserFunding",
    "SensorUpgrade",
    "Transmissions",
    "DeployingCountry",
    "DeploymentComments",
    "ManufactureYear",
    "ManufactureMonth",
    "ManufactureSensorType",
    "ManufactureVoltage",
    "FloatDiameter",
    "SubsfcFloatPresence",
    "DrogueType",
    "DrogueLength",
    "DrogueBallast",
    "DragAreaAboveDrogue",
    "DragAreaOfDrogue",
    "DragAreaRatio",
    "DrogueCenterDepth",
    "DrogueDetectSensor",
]
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


def parse_directory_file(filename: str) -> pd.DataFrame:
    """Read a directory file which contains metadata of drifters' releases.

    Due to naming of these files, a manual update of the last file name after an
    update of the dataset is needed.

    Args:
        filename (str): Name of the directory file
    Returns:
        pd.DataFrame: Sorted list of drifters
    """

    GDP_DIRECTORY_FILE_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/"
    df = pd.read_csv(
        os.path.join(GDP_DIRECTORY_FILE_URL, filename), delimiter="\s+", header=None
    )

    # Combine the date and time columns to easily parse dates below.
    df[4] += " " + df[5]
    df[8] += " " + df[9]
    df[12] += " " + df[13]
    df = df.drop(columns=[5, 9, 13])
    df.columns = [
        "ID",
        "WMO_number",
        "program_number",
        "buoys_type",
        "Deployment_date",
        "Deployment_lat",
        "Deployment_lon",
        "End_date",
        "End_lat",
        "End_lon",
        "Drogue_off_date",
        "death_code",
    ]
    for t in ["Deployment_date", "End_date", "Drogue_off_date"]:
        df[t] = pd.to_datetime(df[t], format="%Y/%m/%d %H:%M", errors="coerce")

    return df


def get_gdp_metadata() -> pd.DataFrame:
    """Download and parse GDP metadata and return it as a Pandas DataFrame."""

    directory_file_names = [
        "dirfl_1_5000.dat",
        "dirfl_5001_10000.dat",
        "dirfl_10001_15000.dat",
        "dirfl_15001_jul22.dat",
    ]
    df = pd.concat([parse_directory_file(f) for f in directory_file_names])
    df.sort_values(["Deployment_date"], inplace=True, ignore_index=True)
    return df


def order_by_date(df: pd.DataFrame, idx: list[int]) -> np.ndarray[int]:
    """From the previously sorted directory files DataFrame, return the drifter
    indices sorted by their end date.

    Args:
        idx [list]: List of drifters to include in the ragged array
    Returns:
        idx [list]: Sorted list of drifters
    """
    return df.ID[np.where(np.in1d(df.ID, idx))[0]].values


def fetch_netcdf(url: str, file: str):
    """Download and save the file from the given url, if not already downloaded."""
    if not os.path.isfile(file):
        req = urllib.request.urlretrieve(url, file)


def download(drifter_ids: list = None, n_random_id: int = None):
    """Download individual NetCDF files from the AOML server.

    :param drifter_ids [list]: list of drifter to retrieve (Default: all)
    :param n_random_id [int]: randomly select n drifter NetCDF files
    :return drifters_ids [list]: list of retrived drifter
    """

    # Create a temporary directory if doesn't already exists.
    os.makedirs(GDP_TMP_PATH, exist_ok=True)

    # retrieve all drifter ID numbers
    if drifter_ids is None:
        urlpath = urllib.request.urlopen(GDP_DATA_URL)
        string = urlpath.read().decode("utf-8")
        pattern = re.compile("drifter_[0-9]*.nc")
        filelist = pattern.findall(string)
        drifter_ids = np.unique([int(f.split("_")[-1][:-3]) for f in filelist])

    # retrieve only a subset of n_random_id trajectories
    if n_random_id:
        if n_random_id > len(drifter_ids):
            warnings.warn(
                f"Retrieving all listed trajectories because {n_random_id} is larger than the {len(drifter_ids)} listed trajectories."
            )
        else:
            rng = np.random.RandomState(42)
            drifter_ids = sorted(rng.choice(drifter_ids, n_random_id, replace=False))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # create list of urls and paths
        urls = []
        files = []
        for i in drifter_ids:
            file = GDP_FILENAME_PATTERN.format(id=i)
            urls.append(os.path.join(GDP_DATA_URL, file))
            files.append(os.path.join(GDP_TMP_PATH, file))

        # parallel retrieving of individual netCDF files
        list(
            tqdm(
                executor.map(fetch_netcdf, urls, files),
                total=len(files),
                desc="Downloading files",
                ncols=80,
            )
        )

    # Download the metadata so we can order the drifter IDs by end date.
    gdp_metadata = get_gdp_metadata()

    return order_by_date(gdp_metadata, drifter_ids)


def decode_date(t):
    """The date format is specified in 'seconds since 1970-01-01 00:00:00' but
    the missing values are stored as -1e+34 which is not supported by the
    default parsing mechanism in xarray.

    This function returns replaced the missing value by NaN and returns a
    datetime instance.
    :param t: date
    :return: datetime
    """
    nat_index = np.logical_or(np.isclose(t, -1e34), np.isnan(t))
    t[nat_index] = np.nan
    return t


def fill_values(var, default=np.nan):
    """Change fill values (-1e+34, inf, -inf) in var array to the value
    specified by default.
    """
    missing_value = np.logical_or(np.isclose(var, -1e34), ~np.isfinite(var))
    if np.any(missing_value):
        var[missing_value] = default
    return var


def str_to_float(value: str, default=np.nan) -> float:
    """Convert a string to float, while returning the value of default if the
    string is not convertible to a float, or if it's a NaN.

    :param value: str
    :return: bool
    """
    try:
        fvalue = float(value)
        if np.isnan(fvalue):
            return default
        else:
            return fvalue
    except ValueError:
        return default


def cut_str(value: str, max_length: int) -> np.chararray:
    """Cut a string to a specific length and return it as a numpy chararray.

    Args:
        value (str): String to cut
        max_length (int): Length of the output
    Returns:
        out (np.chararray): String with max_length characters
    """
    charar = np.chararray(1, max_length)
    charar[:max_length] = value
    return charar


def drogue_presence(lost_time, time):
    """Create drogue status from the drogue lost time and the trajectory time.

    Args:
        lost_time: Timestamp of the drogue loss (or NaT)
        time: Observation time
    Returns:
        out (bool): True if drogues and False otherwise
    """
    if pd.isnull(lost_time) or lost_time >= time[-1]:
        return np.ones_like(time, dtype="bool")
    else:
        return time < lost_time


def rowsize(index: int) -> int:
    return xr.open_dataset(
        os.path.join(GDP_TMP_PATH, GDP_FILENAME_PATTERN.format(id=index)),
        decode_cf=False,
        decode_times=False,
        concat_characters=False,
        decode_coords=False,
    ).dims["obs"]


def preprocess(index: int) -> xr.Dataset:
    """Extract and preprocess the Lagrangian data and attributes. This function
    takes an identification number that can be used to: create a file or url
    pattern or select data from a Dataframe. It then preprocess the data and
    returns a clean xarray Dataset.

    :param index: drifter's identification number
    :return: xr.Dataset containing the data and attributes
    """
    ds = xr.load_dataset(
        os.path.join(GDP_TMP_PATH, GDP_FILENAME_PATTERN.format(id=index)),
        decode_times=False,
        decode_coords=False,
    )

    # parse the date with custom function
    ds["deploy_date"].data = decode_date(np.array([ds.deploy_date.data[0]]))
    ds["end_date"].data = decode_date(np.array([ds.end_date.data[0]]))
    ds["drogue_lost_date"].data = decode_date(np.array([ds.drogue_lost_date.data[0]]))
    ds["time"].data = decode_date(np.array([ds.time.data[0]]))

    # convert fill values to nan
    ds["err_lon"].data = fill_values(ds["err_lon"].data)
    ds["err_lat"].data = fill_values(ds["err_lat"].data)
    ds["err_ve"].data = fill_values(ds["err_ve"].data)
    ds["err_vn"].data = fill_values(ds["err_vn"].data)
    ds["sst"].data = fill_values(ds["sst"].data)
    ds["sst1"].data = fill_values(ds["sst1"].data)
    ds["sst2"].data = fill_values(ds["sst2"].data)
    ds["err_sst"].data = fill_values(ds["err_sst"].data)
    ds["err_sst1"].data = fill_values(ds["err_sst1"].data)
    ds["err_sst2"].data = fill_values(ds["err_sst2"].data)

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
        ds[var].encoding["missing value"] = -1e-34

    # convert type of some variable
    ds["ID"].data = ds["ID"].data.astype("int64")
    ds["WMO"].data = ds["WMO"].data.astype("int32")
    ds["expno"].data = ds["expno"].data.astype("int32")
    ds["typedeath"].data = ds["typedeath"].data.astype("int8")
    ds["flg_sst"].data = ds["flg_sst"].data.astype("int8")
    ds["flg_sst1"].data = ds["flg_sst1"].data.astype("int8")
    ds["flg_sst2"].data = ds["flg_sst2"].data.astype("int8")

    # new variables
    ds["ids"] = (["traj", "obs"], [np.repeat(ds.ID.values, ds.dims["obs"])])
    ds["drogue_status"] = (
        ["traj", "obs"],
        [drogue_presence(ds.drogue_lost_date.data, ds.time.data[0])],
    )

    # convert attributes to variable
    ds["location_type"] = (
        ("traj"),
        [False if ds.location_type == "Argos" else True],
    )  # 0 for Argos, 1 for GPS
    ds["DeployingShip"] = (("traj"), cut_str(ds.DeployingShip, 20))
    ds["DeploymentStatus"] = (("traj"), cut_str(ds.DeploymentStatus, 20))
    ds["BuoyTypeManufacturer"] = (("traj"), cut_str(ds.BuoyTypeManufacturer, 20))
    ds["BuoyTypeSensorArray"] = (("traj"), cut_str(ds.BuoyTypeSensorArray, 20))
    ds["CurrentProgram"] = (("traj"), np.int32([str_to_float(ds.CurrentProgram, -1)]))
    ds["PurchaserFunding"] = (("traj"), cut_str(ds.PurchaserFunding, 20))
    ds["SensorUpgrade"] = (("traj"), cut_str(ds.SensorUpgrade, 20))
    ds["Transmissions"] = (("traj"), cut_str(ds.Transmissions, 20))
    ds["DeployingCountry"] = (("traj"), cut_str(ds.DeployingCountry, 20))
    ds["DeploymentComments"] = (
        ("traj"),
        cut_str(ds.DeploymentComments.encode("ascii", "ignore").decode("ascii"), 20),
    )  # remove non ascii char
    ds["ManufactureYear"] = (("traj"), np.int16([str_to_float(ds.ManufactureYear, -1)]))
    ds["ManufactureMonth"] = (
        ("traj"),
        np.int16([str_to_float(ds.ManufactureMonth, -1)]),
    )
    ds["ManufactureSensorType"] = (("traj"), cut_str(ds.ManufactureSensorType, 20))
    ds["ManufactureVoltage"] = (
        ("traj"),
        np.int16([str_to_float(ds.ManufactureVoltage[:-6], -1)]),
    )  # e.g. 56 V
    ds["FloatDiameter"] = (
        ("traj"),
        [str_to_float(ds.FloatDiameter[:-3])],
    )  # e.g. 35.5 cm
    ds["SubsfcFloatPresence"] = (
        ("traj"),
        np.array([str_to_float(ds.SubsfcFloatPresence)], dtype="bool"),
    )
    ds["DrogueType"] = (("traj"), cut_str(ds.DrogueType, 7))
    ds["DrogueLength"] = (("traj"), [str_to_float(ds.DrogueLength[:-2])])  # e.g. 4.8 m
    ds["DrogueBallast"] = (
        ("traj"),
        [str_to_float(ds.DrogueBallast[:-3])],
    )  # e.g. 1.4 kg
    ds["DragAreaAboveDrogue"] = (
        ("traj"),
        [str_to_float(ds.DragAreaAboveDrogue[:-4])],
    )  # 10.66 m^2
    ds["DragAreaOfDrogue"] = (
        ("traj"),
        [str_to_float(ds.DragAreaOfDrogue[:-4])],
    )  # e.g. 416.6 m^2
    ds["DragAreaRatio"] = (("traj"), [str_to_float(ds.DragAreaRatio)])  # e.g. 39.08
    ds["DrogueCenterDepth"] = (
        ("traj"),
        [str_to_float(ds.DrogueCenterDepth[:-2])],
    )  # e.g. 20.0 m
    ds["DrogueDetectSensor"] = (("traj"), cut_str(ds.DrogueDetectSensor, 20))

    # vars attributes
    vars_attrs = {
        "ID": {"long_name": "Global Drifter Program Buoy ID", "units": "-"},
        "longitude": {"long_name": "Longitude", "units": "degrees_east"},
        "latitude": {"long_name": "Latitude", "units": "degrees_north"},
        "time": {"long_name": "Time", "units": "seconds since 1970-01-01 00:00:00"},
        "ids": {
            "long_name": "Global Drifter Program Buoy ID repeated along observations",
            "units": "-",
        },
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
        "end_date": {
            "long_name": "End date and time",
            "units": "seconds since 1970-01-01 00:00:00",
        },
        "end_lon": {"long_name": "End latitude", "units": "degrees_north"},
        "end_lat": {"long_name": "End longitude", "units": "degrees_east"},
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
        "date_created": datetime.now().isoformat(),
        "publisher_name": "GDP Drifter DAC",
        "publisher_email": "aoml.dftr@noaa.gov",
        "publisher_url": "https://www.aoml.noaa.gov/phod/gdp",
        "licence": "freely available",
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
        ds[var].attrs = vars_attrs[var]
    ds.attrs = attrs

    # rename variables
    ds = ds.rename_vars({"longitude": "lon", "latitude": "lat"})

    return ds


def to_raggedarray(
    drifter_ids: Optional[list[int]] = None, n_random_id: Optional[int] = None
) -> RaggedArray:
    """Download and process individual GDP hourly files and return a
    RaggedArray instance with the data.

    Args:
        drifter_ids [list]: list of drifters to retrieve (Default: all)
        n_random_id [int]: randomly select n drifter NetCDF files
    Returns:
        out [RaggedArray]: A RaggedArray instance of the requested dataset
    """

    ids = download(drifter_ids, n_random_id)

    return RaggedArray.from_files(
        indices=ids,
        preprocess_func=preprocess,
        name_coords=GDP_COORDS,
        name_meta=GDP_METADATA,
        name_data=GDP_DATA,
        rowsize_func=rowsize,
    )

"""
This module provides functions and metadata to convert the Global Drifter
Program (GDP) data to a ``clouddrift.RaggedArray`` instance. The functions
defined in this module are common to both hourly (``clouddrift.adapters.gdp1h``)
and six-hourly (``clouddrift.adapters.gdp6h``) GDP modules.
"""

import os
import re
import tempfile
import urllib.request
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.adapters.utils import download_with_progress, standard_retry_protocol
from clouddrift.raggedarray import DimNames

GDP_DIMS: dict[str, DimNames] = {"traj": "rows", "obs": "obs"}
GDP_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdp")
os.makedirs(
    GDP_TMP_PATH, exist_ok=True
)  # generate temp directory for hurdat related intermerdiary data

GDP_COORDS = [
    "id",
    "time",
]

GDP_METADATA = [
    "rowsize",
    "WMO",
    "expno",
    "deploy_date",
    "deploy_lat",
    "deploy_lon",
    "start_date",
    "start_lat",
    "start_lon",
    "end_date",
    "end_lat",
    "end_lon",
    "drogue_lost_date",
    "typedeath",
    "typebuoy",
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


def cast_float64_variables_to_float32(
    ds: xr.Dataset, variables_to_skip: list[str] = ["time", "lat", "lon"]
) -> xr.Dataset:
    """Cast all float64 variables except ``variables_to_skip`` to float32.
    Extra precision from float64 is not needed and takes up memory and disk
    space.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to modify
    variables_to_skip : list[str]
        List of variables to skip; default is ["time", "lat", "lon"].

    Returns
    -------
    ds : xr.Dataset
        Modified dataset
    """
    for var in ds.variables:
        if var in variables_to_skip:
            continue
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")
    return ds


def parse_directory_file(
    filename: str, tmp_path: str, skip_download: bool = False
) -> pd.DataFrame:
    """Read a GDP directory file that contains metadata of drifter releases.

    Parameters
    ----------
    filename : str
        Name of the directory file to parse.
    tmp_path : str
        Directory where the file is cached locally.
    skip_download : bool, optional
        If True, skip re-downloading the file if it already exists in
        ``tmp_path``. Default is False.

    Returns
    -------
    df : pd.DataFrame
        List of drifters from a single directory file as a pandas DataFrame.
    """
    path = os.path.join(tmp_path, filename)
    gdp_dir_url = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata"
    download_with_progress(
        [(f"{gdp_dir_url}/{filename}", path)], skip_download=skip_download
    )
    df = pd.read_csv(path, delimiter=r"\s+", header=None)
    df[4] += " " + df[5]
    df[8] += " " + df[9]
    df[12] += " " + df[13]
    df = df.drop(columns=[5, 9, 13])
    df.columns = pd.Index(
        [
            "ID",
            "WMO_number",
            "program_number",
            "buoys_type",
            "Start_date",
            "Start_lat",
            "Start_lon",
            "End_date",
            "End_lat",
            "End_lon",
            "Drogue_off_date",
            "death_code",
        ],
        dtype="str",
    )
    for t in ["Start_date", "End_date", "Drogue_off_date"]:
        df[t] = pd.to_datetime(df[t], format="%Y/%m/%d %H:%M", errors="coerce")
    return df


def get_gdp_metadata(
    tmp_path: str = GDP_TMP_PATH,
    skip_download: bool = False,
) -> pd.DataFrame:
    """Download (or read locally cached) GDP metadata and return it as a Pandas DataFrame.

    Parameters
    ----------
    tmp_path : str
        Directory where ``dirfl_*.dat`` metadata files are stored.
    skip_download : bool, optional
        If True, scan ``tmp_path`` for cached ``dirfl_*.dat`` files instead of
        fetching the file list from the network, and skip re-downloading any
        ``dirfl_*.dat`` files that already exist locally. Use this when the
        drifter data files have also been downloaded with ``skip_download=True``
        to ensure the metadata is consistent with the local files.

    Returns
    -------
    df : pd.DataFrame
        Sorted list of drifters as a pandas DataFrame.
    """
    if skip_download:
        metadata_pattern = re.compile(r"dirfl_(\d+)_(\d+|current)\.dat$")
        metadata_files: list[tuple[int, int, str]] = []
        for filename in os.listdir(tmp_path):
            match = metadata_pattern.fullmatch(filename)
            if match is None:
                continue
            low = int(match.group(1))
            high = (
                int(match.group(2))
                if match.group(2) != "current"
                else np.iinfo(int).max
            )
            metadata_files.append((low, high, filename))

        if not metadata_files:
            raise RuntimeError(
                f"No GDP metadata files (dirfl_*.dat) found in {tmp_path}."
            )

        dfs = []
        for _, _, filename in sorted(metadata_files, key=lambda x: (x[0], x[1])):
            try:
                dfs.append(parse_directory_file(filename, tmp_path, skip_download=True))
            except Exception as exc:
                warnings.warn(
                    f"Could not parse local GDP metadata file {filename}; skipping it. Error: {exc}"
                )
        if not dfs:
            raise RuntimeError(
                f"Found GDP metadata files in {tmp_path} but could not parse any of them."
            )
    else:
        directory_files = _list_gdp_directory_files()
        if not directory_files:
            raise RuntimeError("Could not discover GDP metadata directory files.")
        dfs = [
            parse_directory_file(name, tmp_path, skip_download=False)
            for name in directory_files
        ]

    df = pd.concat(dfs)
    df.sort_values(["Start_date"], inplace=True, ignore_index=True)
    return df


def _list_gdp_directory_files() -> list[str]:
    """Fetch the list of GDP metadata directory filenames from the AOML FTP server.

    Returns
    -------
    files : list[str]
        Sorted list of ``dirfl_*.dat`` filenames available on the server.
    """
    gdp_dir_url = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata"
    pattern = re.compile(r"dirfl_(\d+)_(\d+|current)\.dat")

    def _fetch_directory_listing() -> bytes:
        with urllib.request.urlopen(gdp_dir_url) as response:
            return response.read()

    payload = standard_retry_protocol(_fetch_directory_listing)()
    listing = payload.decode("utf-8")

    matches = set(pattern.findall(listing))
    files = [f"dirfl_{low}_{high}.dat" for low, high in matches]

    def _sort_key(name: str) -> tuple[int, int]:
        parsed = pattern.fullmatch(name)
        if parsed is None:
            raise ValueError(f"Unexpected GDP metadata filename: {name}")
        low, high = parsed.groups()
        high_val = int(high) if high != "current" else np.iinfo(int).max
        return int(low), high_val

    return sorted(files, key=_sort_key)


def _subsample(items: list, n: int) -> list:
    if n > len(items):
        warnings.warn(
            f"Requested sample size ({n}) exceeds the number of available trajectories; returning all {len(items)}."
        )
        return items
    rng = np.random.Generator(np.random.MT19937(42))
    return list(rng.choice(items, n, replace=False))


def order_by_date(df: pd.DataFrame, idx: list[int]) -> list[int]:  # noqa: F821
    """From the previously sorted DataFrame of directory files, return the
    unique set of drifter IDs sorted by their start date (the date of the first
    quality-controlled data point).

    Parameters
    ----------
    idx : list
        List of drifters to include in the ragged array

    Returns
    -------
    idx : list
        Unique set of drifter IDs sorted by their start date.
    """
    return df.ID[np.where(np.isin(df.ID, idx))[0]].values  # type: ignore


def fetch_netcdf(url: str, file: str):
    """Download and save the file from the given url, if not already downloaded.

    Parameters
    ----------
    url : str
        URL from which to download the file.
    file : str
        Name of the file to save.
    """
    download_with_progress([(url, file)])


def decode_date(t):
    """The date format is specified as 'seconds since 1970-01-01 00:00:00' but
    the missing values are stored as -1e+34 which is not supported by the
    default parsing mechanism in xarray.

    This function returns replaced the missing value by NaN and returns a
    datetime instance.

    Parameters
    ----------
    t : array
        Array of time values

    Returns
    -------
    out : datetime
        Datetime instance with the missing value replaced by NaN
    """
    nat_index = np.logical_or(np.isclose(t, -1e34), np.isnan(t))
    t[nat_index] = np.nan
    return t


def fill_values(var, default=np.nan):
    """Change fill values (-1e+34, inf, -inf) in var array to the value
    specified by default.

    Parameters
    ----------
    var : array
        Array to fill
    default : float
        Default value to use for fill values
    """
    missing_value = np.logical_or(np.isclose(var, -1e34), ~np.isfinite(var))
    if np.any(missing_value):
        var[missing_value] = default
    return var


def str_to_float(value: str, default: float = np.nan) -> float:
    """Convert a string to float, while returning the value of default if the
    string is not convertible to a float, or if it's a NaN.

    Parameters
    ----------
    value : str
        String to convert to float
    default : float
        Default value to return if the string is not convertible to float

    Returns
    -------
    out : float
        Float value of the string, or default if the string is not convertible to float.
    """
    try:
        fvalue = float(value)
        if np.isnan(fvalue):
            return default
        else:
            return fvalue
    except ValueError:
        return default


def cut_str(value: str, max_length: int) -> np.ndarray:
    """Cut a string to a specific length and return it as a NumPy array with fixed-length strings.

    Parameters
    ----------
    value : str
        String to cut.
    max_length : int
        Maximum length of the output string.

    Returns
    -------
    out : np.ndarray
        NumPy array containing the truncated string with fixed-length dtype.
    """
    truncated_value = value[:max_length]
    char_array = np.array([truncated_value], dtype=f"<U{max_length}")
    return char_array


def drogue_presence(lost_time, time) -> np.ndarray:
    """Create drogue status from the drogue lost time and the trajectory time.

    Parameters
    ----------
    lost_time
        Timestamp of the drogue loss (or NaT)
    time
        Observation time

    Returns
    -------
    out : bool
        True if drogues and False otherwise
    """
    if pd.isnull(lost_time) or lost_time >= time[-1]:
        return np.ones_like(time, dtype="bool")
    else:
        return time < lost_time


def rowsize(index: int, **kwargs) -> int:
    try:
        return xr.open_dataset(
            os.path.join(
                kwargs["tmp_path"], kwargs["filename_pattern"].format(id=index)
            ),
            decode_cf=False,
            decode_times=False,
            concat_characters=False,
            decode_coords=False,
        ).sizes["obs"]
    except Exception as e:
        print(
            f"Error processing {os.path.join(kwargs['tmp_path'], kwargs['filename_pattern'].format(id=index))}"
        )
        print(str(e))
        return 0

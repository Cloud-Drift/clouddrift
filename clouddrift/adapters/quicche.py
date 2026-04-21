"""
This module defines functions used to adapt the QUICCHE CARTHE dataset as a
ragged-arrays dataset.

The dataset contains CARTHE surface drifter trajectories from the Cape Basin
(South Atlantic) in March 2023. The data is hosted at Zenodo in record 14902851.

Example
-------
>>> from clouddrift.adapters import quicche
>>> ds = quicche.to_xarray()
>>> ds = quicche.to_xarray(version="qc1")
>>> ds = quicche.to_xarray(version="raw")

Reference
---------
Zenodo record 14902851: CARTHE surface drifter trajectories, Cape Basin, South Atlantic, March 2023.
"""

import os
import tempfile
import zipfile
from datetime import datetime
from typing import Literal

import pandas as pd
import xarray as xr

from clouddrift.adapters.utils import download_with_progress

# Zenodo record and URL
QUICCHE_ZENODO_RECORD = "14902851"
QUICCHE_URL = (
    "https://zenodo.org/records/14902851/files/CARTHE_Drifters_NSF_QUICCHE.zip"
)
QUICCHE_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "quicche")


def to_xarray(
    version: Literal["raw", "qc1", "qc2", "qc3"] = "qc3",
    tmp_path: str | None = None,
) -> xr.Dataset:
    """
    Parse and convert QUICCHE CARTHE drifter data to an xarray Dataset.

    Parameters
    ----------
    version : Literal["raw", "qc1", "qc2", "qc3"], optional
        Which quality control level to return. "raw" = original raw messages,
        "qc1" = raw data with pre-deployment GPS tests flagged,
        "qc2" = bad records removed,
        "qc3" = QC2 interpolated on a regular 30-minute time grid.
        Default is "qc3".
    tmp_path : str, optional
        Temporary path where intermediary files are stored. If None, uses the default
        temp path defined in this module.

    Returns
    -------
    xarray.Dataset
        QUICCHE CARTHE drifter trajectories as a ragged array with dimensions
        (traj, obs) and coordinates (id, time).
    """
    if tmp_path is None:
        tmp_path = QUICCHE_TMP_PATH
    os.makedirs(tmp_path, exist_ok=True)

    # Validate version
    if version not in ("raw", "qc1", "qc2", "qc3"):
        raise ValueError(
            f"Invalid version '{version}'. Must be one of: raw, qc1, qc2, qc3."
        )

    # Download and extract zip file
    local_zip = f"{tmp_path}/CARTHE_Drifters_NSF_QUICCHE.zip"
    download_with_progress([(QUICCHE_URL, local_zip)])

    # Extract the requested QC file
    if version == "raw":
        target_filename = "quicche_spot_xml_data.dat"
    else:
        target_filename = f"quicche_spot_xml_data_{version}.dat"
    extracted_file = _extract_qc_file(local_zip, target_filename, tmp_path)

    # Parse the data file
    df = _parse_quicche_data(extracted_file, version)

    # Convert to ragged array xarray Dataset
    ds = _dataframe_to_ragged_xarray(df, version)

    return ds


def _extract_qc_file(zip_path: str, target_filename: str, extract_path: str) -> str:
    """
    Extract a specific QC data file from the zip archive.

    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    target_filename : str
        Filename to extract (e.g., 'quicche_spot_xml_data_qc2.dat').
    extract_path : str
        Directory to extract files to.

    Returns
    -------
    str
        Full path to the extracted file.

    Raises
    ------
    FileNotFoundError
        If the target file is not found in the zip archive.
    """
    extracted_file = os.path.join(extract_path, target_filename)

    # Only extract if not already present
    if not os.path.exists(extracted_file):
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the file in the archive (may be nested in a subdirectory)
            matching_files = [f for f in zf.namelist() if f.endswith(target_filename)]

            if not matching_files:
                available_files = [f for f in zf.namelist() if f.endswith(".dat")]
                raise FileNotFoundError(
                    f"Could not find '{target_filename}' in zip archive. "
                    f"Available .dat files: {available_files}"
                )

            # Extract the first match
            file_in_zip = matching_files[0]
            with zf.open(file_in_zip) as source, open(extracted_file, "wb") as target:
                target.write(source.read())

    return extracted_file


def _parse_quicche_data(
    filepath: str,
    version: Literal["raw", "qc1", "qc2", "qc3"],
) -> pd.DataFrame:
    """
    Parse a QUICCHE CARTHE data file into a pandas DataFrame.

    The file is whitespace-delimited with 9-10 columns:
    1. manufacturer_message_id
    2. manufacturer_gps_id
    3. drifter_id
    4. time (ISO 8601 format: YYYY-MM-DDTHH:mm:ss.SSSZ)
    5. manufacturer_time_seconds
    6. latitude (decimal degrees North)
    7. longitude (decimal degrees East)
    8. gps_record_setting
    9. battery_state
    10. predeployment_flag (optional, may be empty)

    Only columns 3, 4, 6, 7 (drifter_id, time, latitude, longitude) are
    required for trajectory definition.

    Parameters
    ----------
    filepath : str
        Path to the .dat file to parse.

    version : Literal["raw", "qc1", "qc2", "qc3"]
        QUICCHE processing level to parse.

    Returns
    -------
    pd.DataFrame
        Parsed dataframe containing trajectory columns and version-specific
        observation metadata columns.
    """
    col_names = [
        "manufacturer_message_id",
        "manufacturer_gps_id",
        "drifter_id",
        "time_iso8601",
        "manufacturer_time_seconds",
        "latitude",
        "longitude",
        "gps_record_setting",
        "battery_state",
        "predeployment_flag",
    ]

    # Read the file, allowing for 9-10 columns
    df = pd.read_csv(
        filepath,
        names=col_names,
        sep=r"\s+",
        header=None,
        engine="python",
        dtype={
            "drifter_id": str,
            "time_iso8601": str,
            "latitude": float,
            "longitude": float,
        },
    )

    # Normalize optional/string fields for robust NetCDF serialization.
    df["battery_state"] = df["battery_state"].fillna("").astype(str)
    df["predeployment_flag"] = df["predeployment_flag"].fillna("").astype(str)

    selected_columns = ["drifter_id", "time_iso8601", "latitude", "longitude"]
    if version in ("raw", "qc1", "qc2"):
        selected_columns.append("battery_state")
    if version == "qc1":
        selected_columns.append("predeployment_flag")

    df = df[selected_columns].copy()

    # Parse time as UTC then drop timezone info so NetCDF serialization uses
    # plain datetime64[ns] instead of Python-object timestamps.
    parsed_time = pd.to_datetime(df["time_iso8601"], utc=True, errors="coerce")
    df["time"] = parsed_time.dt.tz_localize(None)

    # Sort by drifter_id and time
    df = df.sort_values(["drifter_id", "time"]).reset_index(drop=True)

    # Remove the ISO string column, keep the parsed time and selected metadata.
    ordered_columns = ["drifter_id", "time", "latitude", "longitude"]
    if "battery_state" in df.columns:
        ordered_columns.append("battery_state")
    if "predeployment_flag" in df.columns:
        ordered_columns.append("predeployment_flag")

    df = df[ordered_columns].copy()
    if "predeployment_flag" in df.columns:
        df = df.rename(columns={"predeployment_flag": "flag"})

    return df


def _dataframe_to_ragged_xarray(df: pd.DataFrame, version: str) -> xr.Dataset:
    """
    Convert a trajectory DataFrame to a ragged array xarray Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: drifter_id, time, latitude, longitude.
    version : str
        Version ("raw", "qc1", "qc2", or "qc3") for metadata.

    Returns
    -------
    xr.Dataset
        Ragged array dataset with dimensions (traj, obs).
    """
    # Compute rowsize and unique IDs
    rowsize_series = df.groupby("drifter_id", sort=True).size()
    unique_ids = rowsize_series.index.to_numpy()
    rowsize = rowsize_series.to_numpy(dtype="int64")
    # Replicate drifter_id for each observation row
    ds = xr.Dataset.from_dataframe(df)

    # Rename the implicit dataframe index dimension to obs and drop the
    # redundant coordinate variable that only mirrors the observation index.
    ds = ds.rename_dims({"index": "obs"})

    # Assign trajectory-level coordinates
    ds = ds.assign({"id": ("traj", unique_ids)})
    ds = ds.assign({"rowsize": ("traj", rowsize)})

    # Drop per-observation drifter_id and the redundant dataframe index coordinate.
    ds = ds.drop_vars(["drifter_id", "index"])

    # Set coordinates
    ds = ds.set_coords(["id", "time"])

    # Cast dtypes to project conventions
    ds["latitude"] = ds["latitude"].astype("float32")
    ds["longitude"] = ds["longitude"].astype("float32")
    ds["rowsize"] = ds["rowsize"].astype("int64")

    # Define variable attributes
    var_attrs = {
        "latitude": {
            "long_name": "Latitude of drifter position",
            "units": "degrees_north",
        },
        "longitude": {
            "long_name": "Longitude of drifter position",
            "units": "degrees_east",
        },
        "time": {
            "long_name": "Time of observation",
            "comment": "UTC timestamps parsed from source ISO 8601 strings ending with 'Z'",
        },
        "id": {
            "long_name": "Drifter ID",
            "units": "-",
        },
        "rowsize": {
            "long_name": "Number of observations per trajectory",
            "units": "-",
        },
        "battery_state": {
            "long_name": "Battery state reported by manufacturer",
            "comment": "Values include GOOD and LOW",
            "units": "-",
        },
        "flag": {
            "long_name": "QC1 position/test flag",
            "comment": "PRE: pre-deployment test; BAD_POS: visually evaluated bad position; empty string: no issue",
            "units": "-",
        },
    }

    # Apply variable attributes
    for var, attrs in var_attrs.items():
        if var in ds.variables:
            ds[var].attrs = attrs

    # Define global attributes
    qc_description = {
        "raw": "raw data",
        "qc1": "raw data with pre-deployment GPS tests flagged",
        "qc2": "bad records removed",
        "qc3": "QC2 interpolated on a regular 30 minute time grid",
    }

    global_attrs = {
        "title": f"QUICCHE CARTHE Surface Drifter Trajectories ({version.upper()})",
        "summary": f"CARTHE surface drifter trajectories from the Cape Basin (South Atlantic), March 2023. QC level {version.upper()}: {qc_description[version]}",
        "source": "CARTHE surface drifters",
        "time_zone": "UTC",
        "date_created": datetime.now().isoformat(),
        "history": f"Dataset downloaded from Zenodo record {QUICCHE_ZENODO_RECORD}; processed on {datetime.now().strftime('%Y-%m-%d')}",
        "publisher_name": "Zenodo",
        "publisher_url": f"https://zenodo.org/records/{QUICCHE_ZENODO_RECORD}",
        "Conventions": "CF-1.6",
        "featureType": "trajectory",
        "qc_level": version,
    }

    ds.attrs = global_attrs

    return ds

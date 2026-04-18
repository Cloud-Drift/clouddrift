"""
This module defines functions used to adapt the Cape Basin CARTHE dataset as a
ragged-arrays dataset.

The dataset contains CARTHE surface drifter trajectories from the Cape Basin
(South Atlantic) in March 2023. The data is hosted at Zenodo in record 14902851.

Example
-------
>>> from clouddrift.adapters import cape_basin
>>> ds = cape_basin.to_xarray()
>>> ds = cape_basin.to_xarray(version='qc2')

Reference
---------
Zenodo record 14902851: CARTHE surface drifter trajectories, Cape Basin, South Atlantic, March 2023.
"""

import os
import tempfile
import warnings
import zipfile
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.adapters.utils import download_with_progress

# Zenodo record and URL
CAPE_BASIN_ZENODO_RECORD = "14902851"
CAPE_BASIN_URL = "https://zenodo.org/records/14902851/files/CARTHE_Drifters_NSF_QUICCHE.zip"
CAPE_BASIN_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "cape_basin")
CAPE_BASIN_VERSION = "2026-04"


def to_xarray(
    version: Literal["qc2", "qc3"] = "qc3",
    tmp_path: str | None = None,
) -> xr.Dataset:
    """
    Parse and convert Cape Basin CARTHE drifter data to an xarray Dataset.

    Parameters
    ----------
    version : Literal["qc2", "qc3"], optional
        Which quality control level to return. "qc2" = bad records removed;
        "qc3" = QC2 interpolated on a regular 30-minute time grid. Default is "qc3".
    tmp_path : str, optional
        Temporary path where intermediary files are stored. If None, uses the default
        temp path defined in this module.

    Returns
    -------
    xarray.Dataset
        Cape Basin CARTHE drifter trajectories as a ragged array with dimensions
        (traj, obs) and coordinates (id, time).
    """
    if tmp_path is None:
        tmp_path = CAPE_BASIN_TMP_PATH
        os.makedirs(tmp_path, exist_ok=True)

    # Validate version
    if version not in ("qc2", "qc3"):
        raise ValueError(
            f"Invalid version '{version}'. Must be 'qc2' or 'qc3'."
        )

    # Download and extract zip file
    local_zip = f"{tmp_path}/CARTHE_Drifters_NSF_QUICCHE.zip"
    download_with_progress([(CAPE_BASIN_URL, local_zip)])

    # Extract the requested QC file
    target_filename = f"quicche_spot_xml_data_{version}.dat"
    extracted_file = _extract_qc_file(local_zip, target_filename, tmp_path)

    # Parse the data file
    df = _parse_cape_basin_data(extracted_file)

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
                available_files = [
                    f for f in zf.namelist() if f.endswith(".dat")
                ]
                raise FileNotFoundError(
                    f"Could not find '{target_filename}' in zip archive. "
                    f"Available .dat files: {available_files}"
                )

            # Extract the first match
            file_in_zip = matching_files[0]
            with zf.open(file_in_zip) as source, open(
                extracted_file, "wb"
            ) as target:
                target.write(source.read())

    return extracted_file


def _parse_cape_basin_data(filepath: str) -> pd.DataFrame:
    """
    Parse a Cape Basin CARTHE data file into a pandas DataFrame.

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

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: drifter_id, time, latitude, longitude.
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

    # Ensure predeployment_flag exists (may be missing in some rows)
    if "predeployment_flag" not in df.columns:
        df["predeployment_flag"] = ""

    # Extract only required columns for trajectory
    df = df[["drifter_id", "time_iso8601", "latitude", "longitude"]].copy()

    # Parse time as UTC then drop timezone info so NetCDF serialization uses
    # plain datetime64[ns] instead of Python-object timestamps.
    parsed_time = pd.to_datetime(df["time_iso8601"], utc=True, errors="coerce")
    df["time"] = parsed_time.dt.tz_localize(None)

    # Sort by drifter_id and time
    df = df.sort_values(["drifter_id", "time"]).reset_index(drop=True)

    # Remove the ISO string column, keep the parsed time
    df = df[["drifter_id", "time", "latitude", "longitude"]].copy()

    return df


def _dataframe_to_ragged_xarray(df: pd.DataFrame, version: str) -> xr.Dataset:
    """
    Convert a trajectory DataFrame to a ragged array xarray Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: drifter_id, time, latitude, longitude.
    version : str
        QC version ("qc2" or "qc3") for metadata.

    Returns
    -------
    xr.Dataset
        Ragged array dataset with dimensions (traj, obs).
    """
    # Get unique drifter IDs as fixed-width unicode (avoid object dtype in NetCDF)
    drifter_ids = df["drifter_id"].astype(str).to_numpy()
    unique_ids, rowsize = np.unique(drifter_ids, return_counts=True)

    # Create the ragged array dataset
    # Replicate drifter_id for each observation row
    ds = xr.Dataset.from_dataframe(df)

    # Rename index dimension to obs
    ds = ds.rename_dims({"index": "obs"})

    # Assign trajectory-level coordinates
    ds = ds.assign({"id": ("traj", unique_ids)})
    ds = ds.assign({"rowsize": ("traj", rowsize)})

    # Drop the per-observation drifter_id column (keep only traj-level id)
    ds = ds.drop_vars(["drifter_id"])

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
    }

    # Apply variable attributes
    for var, attrs in var_attrs.items():
        if var in ds.variables:
            ds[var].attrs = attrs

    # Define global attributes
    qc_description = {
        "qc2": "bad records removed",
        "qc3": "QC2 interpolated on a regular 30 minute time grid",
    }

    global_attrs = {
        "title": f"Cape Basin CARTHE Surface Drifter Trajectories ({version.upper()})",
        "summary": f"CARTHE surface drifter trajectories from the Cape Basin (South Atlantic), March 2023. QC level {version.upper()}: {qc_description[version]}",
        "source": "CARTHE surface drifters",
        "time_zone": "UTC",
        "date_created": datetime.now().isoformat(),
        "history": f"Dataset downloaded from Zenodo record {CAPE_BASIN_ZENODO_RECORD}; processed on {datetime.now().strftime('%Y-%m-%d')}",
        "publisher_name": "Zenodo",
        "publisher_url": f"https://zenodo.org/records/{CAPE_BASIN_ZENODO_RECORD}",
        "Conventions": "CF-1.6",
        "featureType": "trajectory",
        "qc_level": version,
    }

    ds.attrs = global_attrs

    return ds

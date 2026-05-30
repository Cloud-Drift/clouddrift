"""
This module defines functions used to adapt the QUICCHE CARTHE dataset as a
ragged-arrays dataset.

The dataset contains CARTHE surface drifter trajectories from the Cape Basin
(South Atlantic) in March 2023. The data is hosted at Zenodo in record 14902851.

Example
-------
>>> from clouddrift.adapters import quicche
>>> ra = quicche.to_raggedarray()
>>> ra = quicche.to_raggedarray(version="qc1")
>>> ra = quicche.to_raggedarray(version="raw")

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

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

# Zenodo record and URL
QUICCHE_ZENODO_RECORD = "14902851"
QUICCHE_URL = "https://zenodo.org/records/14902851/files/CARTHE_Drifters_NSF_QUICCHE.zip"
QUICCHE_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "quicche")


def to_raggedarray(
    version: Literal["raw", "qc1", "qc2", "qc3"] = "qc3",
    tmp_path: str | None = None,
    skip_download: bool = False,
) -> RaggedArray:
    """
    Parse and convert QUICCHE CARTHE drifter data to a RaggedArray instance.

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
    skip_download : bool, optional
        If True, skip re-downloading the ZIP file if it already exists in
        ``tmp_path``. Default is False.

    Returns
    -------
    RaggedArray
        QUICCHE CARTHE drifter trajectories as a ragged array with dimensions
        (traj, obs) and coordinates (id, time).
    """
    if tmp_path is None:
        tmp_path = QUICCHE_TMP_PATH
    os.makedirs(tmp_path, exist_ok=True)

    # Validate version
    if version not in ("raw", "qc1", "qc2", "qc3"):
        raise ValueError(f"Invalid version '{version}'. Must be one of: raw, qc1, qc2, qc3.")

    # Download and extract zip file
    local_zip = f"{tmp_path}/CARTHE_Drifters_NSF_QUICCHE.zip"
    download_with_progress([(QUICCHE_URL, local_zip)], skip_download=skip_download)

    # Extract the requested QC file
    if version == "raw":
        target_filename = "quicche_spot_xml_data.dat"
    else:
        target_filename = f"quicche_spot_xml_data_{version}.dat"
    extracted_file = _extract_qc_file(local_zip, target_filename, tmp_path)

    # Parse the data file
    df = _parse_quicche_data(extracted_file, version)

    # Convert to ragged array
    ra = _dataframe_to_raggedarray(df, version)

    return ra


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


def _dataframe_to_raggedarray(df: pd.DataFrame, version: str) -> RaggedArray:
    """
    Convert a trajectory DataFrame to a RaggedArray instance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: drifter_id, time, latitude, longitude.
    version : str
        Version ("raw", "qc1", "qc2", or "qc3") for metadata.

    Returns
    -------
    RaggedArray
        Ragged array with dimensions (traj, obs).
    """
    # Compute rowsize and unique IDs
    rowsize_series = df.groupby("drifter_id", sort=True).size()
    unique_ids = rowsize_series.index.to_numpy()
    rowsize = rowsize_series.to_numpy(dtype="int64")

    qc_description = {
        "raw": "raw data",
        "qc1": "raw data with pre-deployment GPS tests flagged",
        "qc2": "bad records removed",
        "qc3": "QC2 interpolated on a regular 30 minute time grid",
    }

    attrs_global = {
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

    attrs_variables = {
        "id": {"long_name": "Drifter ID", "units": "-"},
        "time": {
            "long_name": "Time of observation",
            "comment": "UTC timestamps parsed from source ISO 8601 strings ending with 'Z'",
        },
        "rowsize": {"long_name": "Number of observations per trajectory", "units": "-"},
        "latitude": {"long_name": "Latitude of drifter position", "units": "degrees_north"},
        "longitude": {"long_name": "Longitude of drifter position", "units": "degrees_east"},
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

    data: dict = {
        "latitude": df["latitude"].to_numpy(dtype="float32"),
        "longitude": df["longitude"].to_numpy(dtype="float32"),
    }
    var_dims: dict = {
        "rowsize": ["traj"],
        "latitude": ["obs"],
        "longitude": ["obs"],
    }

    if "battery_state" in df.columns:
        data["battery_state"] = df["battery_state"].to_numpy()
        var_dims["battery_state"] = ["obs"]

    if "flag" in df.columns:
        data["flag"] = df["flag"].to_numpy()
        var_dims["flag"] = ["obs"]

    return RaggedArray(
        coords={
            "id": unique_ids,
            "time": df["time"].to_numpy(dtype="datetime64[ns]"),
        },
        metadata={
            "rowsize": rowsize,
        },
        data=data,
        attrs_global=attrs_global,
        attrs_variables=attrs_variables,
        name_dims={"traj": "rows", "obs": "obs"},
        coord_dims={"id": "traj", "time": "obs"},
        var_dims=var_dims,
    )

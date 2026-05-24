"""
This module defines functions used to adapt the Grand LAgrangian Deployment
(GLAD) CODE-style drifter trajectories dataset from the Consortium for Advanced Research
on Transport of Hydrocarbon in the Environment (CARTHE) as a ragged-array dataset.

The datasets and their respective description are hosted at:

- raw trajectories https://doi.org/10.7266/N7086388
- qc1: 5-min interpolation, no filtering https://doi.org/10.7266/N7416V0M
- qc2: low-pass filtered, 15 minute interval records https://doi.org/10.7266/N7VD6WC8

Example
-------
>>> from clouddrift.adapters import glad
>>> ra = glad.to_raggedarray()

to retrieve the default `qc2` version of the dataset. To retrieve the `raw` or `qc1` versions, pass the
version name as an argument to `to_xarray()`, e.g. `glad.to_xarray(version="raw")`.

References
----------
Tamay Özgökmen. 2016. Grand Lagrangian Deployment GLAD experiment CODE-style and flat surface drifter trajectories (raw), northern Gulf of Mexico near DeSoto Canyon, July 2012 - January 2013. Distributed by: GRIIDC, Harte Research Institute, Texas A&M University–Corpus Christi. https://doi.org/10.7266/N7086388
Huntley, H. S., Lipphardt, B. L., & Kirwan, A. D. (2019). Anisotropy and Inhomogeneity in Drifter Dispersion. Journal of Geophysical Research: Oceans, 124(12), 8667–8682. doi:10.1029/2019jc015179
Özgökmen, Tamay. 2013. GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records), northern Gulf of Mexico near DeSoto Canyon, July-October 2012. Distributed by: Gulf of Mexico Research Initiative Information and Data Cooperative (GRIIDC), Harte Research Institute, Texas A&M University–Corpus Christi. doi:10.7266/N7VD6WC8
"""

from io import BytesIO
from typing import Literal

import numpy as np
import pandas as pd

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

GLAD_VERSIONS = Literal["raw", "qc1", "qc2"]
URL_RAW = "https://data.griidc.org/api/file/download/163324"
URL_QC1 = "https://data.griidc.org/api/file/download/152756"
URL_QC2 = "https://data.griidc.org/api/file/download/169841"

_DATASET_VERSIONS = {
    # GRIIDC server doesn't provide Content-Length header,
    # so we'll hardcode the expected data length here.
    "raw": (URL_RAW, 296648132),
    "qc1": (URL_QC1, 534489318),
    "qc2": (URL_QC2, 155330876),
}


def get_dataframe(version: GLAD_VERSIONS = "qc2") -> pd.DataFrame:
    """Get a GLAD dataset version as a pandas DataFrame."""
    if version not in _DATASET_VERSIONS:
        raise ValueError(f"Unknown GLAD version '{version}'. Expected one of: raw, qc1, qc2")

    url, file_size = _DATASET_VERSIONS[version]
    buf = BytesIO(b"")
    download_with_progress([(url, buf, file_size)])
    actual_size = len(buf.getbuffer())
    if actual_size < file_size // 2:
        raise ConnectionError(
            f"Downloaded only {actual_size:,} bytes from GLAD data server"
            f" (expected ~{file_size:,}), url={url}"
        )
    buf.seek(0)
    column_names = [
        "id",
        "date",
        "time",
        "latitude",
        "longitude",
        "position_error",
        "u",
        "v",
        "velocity_error",
    ]
    df = pd.read_csv(buf, sep=r"\s+", skiprows=5, names=column_names)
    df["obs"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.drop(["date", "time"], axis=1, inplace=True)
    return df


def to_raggedarray(version: GLAD_VERSIONS = "qc2") -> RaggedArray:
    """Return a GLAD dataset version as a RaggedArray instance.

    Parameters
    ----------
    version : GLAD_VERSIONS, optional
        Version of the GLAD dataset to retrieve. One of "raw", "qc1", "qc2".
        Default is "qc2".

    Returns
    -------
    RaggedArray
        GLAD dataset as a ragged array.
    """
    df = get_dataframe(version=version)

    ids = df["id"].to_numpy()
    traj, rowsize = np.unique(ids, return_counts=True)

    attrs_global = {
        "title": "GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records), northern Gulf of Mexico near DeSoto Canyon, July-October 2012",
        "institution": "Consortium for Advanced Research on Transport of Hydrocarbon in the Environment (CARTHE)",
        "source": "CODE-style drifters",
        "history": "Downloaded from https://data.gulfresearchinitiative.org/data/R1.x134.073:0004 and post-processed into a ragged-array dataset by CloudDrift",
        "references": "Özgökmen, Tamay. 2013. GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records), northern Gulf of Mexico near DeSoto Canyon, July-October 2012. Distributed by: Gulf of Mexico Research Initiative Information and Data Cooperative (GRIIDC), Harte Research Institute, Texas A&M University–Corpus Christi. doi:10.7266/N7VD6WC8",
    }

    attrs_variables = {
        "id": {"long_name": "trajectory identifier"},
        "time": {"long_name": "time"},
        "rowsize": {
            "long_name": "number of observations per trajectory",
            "sample_dimension": "obs",
            "units": "-",
        },
        "longitude": {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        },
        "latitude": {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        },
        "position_error": {
            "long_name": "position_error",
            "units": "m",
        },
        "u": {
            "long_name": "eastward_sea_water_velocity",
            "standard_name": "eastward_sea_water_velocity",
            "units": "m s-1",
        },
        "v": {
            "long_name": "northward_sea_water_velocity",
            "standard_name": "northward_sea_water_velocity",
            "units": "m s-1",
        },
        "velocity_error": {
            "long_name": "velocity_error",
            "units": "m s-1",
        },
    }

    return RaggedArray(
        coords={
            "id": traj,
            "time": df["obs"].to_numpy(dtype="datetime64[ns]"),
        },
        metadata={
            "rowsize": rowsize.astype("int64"),
        },
        data={
            "latitude": df["latitude"].to_numpy(dtype="float32"),
            "longitude": df["longitude"].to_numpy(dtype="float32"),
            "position_error": df["position_error"].to_numpy(dtype="float32"),
            "u": df["u"].to_numpy(dtype="float32"),
            "v": df["v"].to_numpy(dtype="float32"),
            "velocity_error": df["velocity_error"].to_numpy(dtype="float32"),
        },
        attrs_global=attrs_global,
        attrs_variables=attrs_variables,
        name_dims={"traj": "rows", "obs": "obs"},
        coord_dims={"id": "traj", "time": "obs"},
        var_dims={
            "rowsize": ["traj"],
            "latitude": ["obs"],
            "longitude": ["obs"],
            "position_error": ["obs"],
            "u": ["obs"],
            "v": ["obs"],
            "velocity_error": ["obs"],
        },
    )

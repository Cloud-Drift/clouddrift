"""Adapt the LASER 15-minute interpolated drifter trajectories to Xarray.

The upstream dataset is hosted by GRIIDC at https://doi.org/10.7266/N7W0940J
and distributed as a zip archive containing the ASCII drifter trajectories file
and a README.

Example
-------
>>> from clouddrift.adapters import laser
>>> ds = laser.to_xarray()

References
----------
Eric D'Asaro, Cedric Guigand, Angelique Haza, Helga Huntley, Guillaume Novelli,
Tamay Ozgokmen, Ed Ryan. 2017. Lagrangian Submesoscale Experiment (LASER)
surface drifters, interpolated to 15-minute intervals. Distributed by: GRIIDC,
Harte Research Institute, Texas A&M University-Corpus Christi.
https://doi.org/10.7266/N7W0940J
"""

import os
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.adapters.utils import download_with_progress

_DATASET_TITLE = (
    "Lagrangian Submesoscale Experiment (LASER) surface drifters, "
    "interpolated to 15-minute intervals"
)
_DATASET_PAGE = "https://data.griidc.org/data/R4.x265.237:0001"
_DOWNLOAD_URL = "https://data.griidc.org/api/datasets/zip/2101"
_DATA_FILENAME = "laser_spot_drifters_clean_v15.dat"
LASER_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "laser")
_LOCAL_ARCHIVE_NAME = "laser_surface_drifters.zip"


def _open_datafile(archive: ZipFile):
    for member in archive.namelist():
        if member == _DATA_FILENAME or member.endswith(f"/{_DATA_FILENAME}"):
            return archive.open(member)

    raise FileNotFoundError(
        f"Could not find '{_DATA_FILENAME}' in LASER dataset archive."
    )


def get_dataframe(
    tmp_path: str | None = None,
    skip_download: bool = False,
) -> pd.DataFrame:
    """Get the LASER dataset as a pandas DataFrame.

    Parameters
    ----------
    tmp_path : str, optional
        Temporary path where intermediary files are stored. If None, uses the
        default LASER adapter temp path.
    skip_download : bool, optional
        If True, skip re-downloading the archive if it already exists in
        ``tmp_path``. Default is False.
    """
    if tmp_path is None:
        tmp_path = LASER_TMP_PATH
    os.makedirs(tmp_path, exist_ok=True)

    local_zip = os.path.join(tmp_path, _LOCAL_ARCHIVE_NAME)
    download_with_progress([(_DOWNLOAD_URL, local_zip)], skip_download=skip_download)

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

    with ZipFile(local_zip) as archive:
        with _open_datafile(archive) as data_file:
            df = pd.read_csv(
                data_file,
                sep=r"\s+",
                comment="%",
                names=column_names,
            )

    df["obs"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"])
    return df.sort_values(["id", "obs"], kind="stable").reset_index(drop=True)


def to_xarray(
    tmp_path: str | None = None,
    skip_download: bool = False,
) -> xr.Dataset:
    """Return the LASER dataset as a ragged-array Xarray Dataset.

    Parameters
    ----------
    tmp_path : str, optional
        Temporary path where intermediary files are stored. If None, uses the
        default LASER adapter temp path.
    skip_download : bool, optional
        If True, skip re-downloading the archive if it already exists in
        ``tmp_path``. Default is False.
    """
    df = get_dataframe(tmp_path=tmp_path, skip_download=skip_download)
    ds = df.to_xarray()

    traj, rowsize = np.unique(ds.id, return_counts=True)

    ds = (
        ds.swap_dims({"index": "obs"})
        .drop_vars(["id", "index"])
        .assign_coords(traj=traj)
        .assign({"rowsize": ("traj", rowsize)})
        .rename_vars({"obs": "time", "traj": "id"})
    )

    for var in ds.variables:
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")

    ds["longitude"].attrs = {
        "long_name": "longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
    }

    ds["latitude"].attrs = {
        "long_name": "latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
    }

    ds["position_error"].attrs = {
        "long_name": "position_error",
        "units": "m",
    }

    ds["u"].attrs = {
        "long_name": "eastward_sea_water_velocity",
        "standard_name": "eastward_sea_water_velocity",
        "units": "m s-1",
    }

    ds["v"].attrs = {
        "long_name": "northward_sea_water_velocity",
        "standard_name": "northward_sea_water_velocity",
        "units": "m s-1",
    }

    ds["velocity_error"].attrs = {
        "long_name": "velocity_error",
        "units": "m s-1",
    }

    ds.attrs = {
        "title": _DATASET_TITLE,
        "institution": "Consortium for Advanced Research on Transport of Hydrocarbon in the Environment (CARTHE)",
        "source": "SPOT GPS drifters",
        "history": f"Downloaded from {_DATASET_PAGE} and post-processed into a ragged-array Xarray Dataset by CloudDrift",
        "references": "Eric D'Asaro, Cedric Guigand, Angelique Haza, Helga Huntley, Guillaume Novelli, Tamay Ozgokmen, Ed Ryan. 2017. Lagrangian Submesoscale Experiment (LASER) surface drifters, interpolated to 15-minute intervals. Distributed by: GRIIDC, Harte Research Institute, Texas A&M University-Corpus Christi. https://doi.org/10.7266/N7W0940J",
    }

    return ds

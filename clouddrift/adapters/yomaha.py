"""
This module defines functions used to adapt the YoMaHa'07: Velocity data assessed
from trajectories of Argo floats at parking level and at the sea surface as
a ragged-arrays dataset.

The dataset is hosted at http://apdrc.soest.hawaii.edu/projects/yomaha/ and the user manual
is available at http://apdrc.soest.hawaii.edu/projects/yomaha/yomaha07/YoMaHa070612.pdf.

Example
-------
>>> from clouddrift.adapters import yomaha
>>> ra = yomaha.to_raggedarray()

Reference
---------
Lebedev, K. V., Yoshinari, H., Maximenko, N. A., & Hacker, P. W. (2007). Velocity data assessed from trajectories of Argo floats at parking level and at the sea surface. IPRC Technical Note, 4(2), 1-16.
"""

import gzip
import logging
import os
import shutil
import tempfile
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

_logger = logging.getLogger(__name__)
YOMAHA_URLS = [
    # order of the URLs is important
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/float_types.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/DACs.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/0-date_time.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/WMO2DAC2type.txt",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/end-prog.lst",
    "http://apdrc.soest.hawaii.edu/projects/Argo/data/trjctry/0-Near-Real_Time/yomaha07.dat.gz",
]
YOMAHA_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "yomaha")


def download(tmp_path: str, skip_download: bool = False):
    download_requests = [(url, f"{tmp_path}/{url.split('/')[-1]}") for url in YOMAHA_URLS[:-1]]
    download_with_progress(download_requests, skip_download=skip_download)

    filename_gz = f"{tmp_path}/{YOMAHA_URLS[-1].split('/')[-1]}"
    filename = filename_gz.removesuffix(".gz")
    decompressed_fp = os.path.join(tmp_path, filename)

    if not (skip_download and os.path.exists(decompressed_fp)):
        buffer = BytesIO()
        download_with_progress([(YOMAHA_URLS[-1], buffer)])
        if len(buffer.getvalue()) == 0:
            raise ConnectionError(
                f"Downloaded invalid data from Yomaha data server (url={YOMAHA_URLS[-1]})"
            )
        buffer.seek(0)
        with (
            open(decompressed_fp, "wb") as file,
            gzip.open(buffer, "rb") as compressed_file,
        ):
            shutil.copyfileobj(compressed_file, file)


def to_raggedarray(tmp_path: str | None = None, skip_download: bool = False) -> RaggedArray:
    """Convert the YoMaHa'07 dataset to a RaggedArray instance.

    Parameters
    ----------
    tmp_path : str, optional
        Path where the dataset files are cached. Defaults to a platform-specific
        temporary directory.
    skip_download : bool, optional
        If True, skip re-downloading files that already exist in ``tmp_path``.
        The main data file (``yomaha07.dat.gz``) is skipped when its
        decompressed version already exists locally. Default is False.

    Returns
    -------
    RaggedArray
        YoMaHa'07 dataset as a ragged array.
    """
    if tmp_path is None:
        tmp_path = YOMAHA_TMP_PATH
    os.makedirs(tmp_path, exist_ok=True)

    # get or update required files
    download(tmp_path, skip_download=skip_download)

    # database last update
    with open(f"{tmp_path}/{YOMAHA_URLS[2].split('/')[-1]}") as f:
        YOMAHA_VERSION = f.read().strip()
        print(f"Last database update was: {YOMAHA_VERSION}")

    # parse with pandas
    col_names = [
        "lon_d",
        "lat_d",
        "pres_d",
        "time_d",
        "ve_d",
        "vn_d",
        "err_ve_d",
        "err_vn_d",
        "lon_s",
        "lat_s",
        "time_s",
        "ve_s",
        "vn_s",
        "err_ve_s",
        "err_vn_s",
        "lon_lp",
        "lat_lp",
        "time_lp",
        "lon_fc",
        "lat_fc",
        "time_fc",
        "lon_lc",
        "lat_lc",
        "time_lc",
        "surf_fix",
        "id",
        "cycle",
        "time_inv",
    ]

    na_col = list(
        map(
            lambda x: str(x),
            [
                -999.9999,
                -99.9999,
                -999.9,
                -999.999,
                -999.99,
                -999.99,
                -999.99,
                -999.99,
                -999.99,
                -99.99,
                -999.99,
                -999.99,
                -999.99,
                -999.99,
                -999.99,
                -999.99,
                -99.99,
                -999.99,
                -999.99,
                -99.99,
                -999.99,
                -999.99,
                -99.99,
                -999.99,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        )
    )

    filename_gz = f"{tmp_path}/{YOMAHA_URLS[-1].split('/')[-1]}"
    filename = filename_gz.removesuffix(".gz")
    df = pd.read_csv(filename, names=col_names, sep=r"\s+", header=None, na_values=na_col)

    unique_id, rowsize = np.unique(df["id"], return_counts=True)

    # mapping of yomaha float id, wmo float id, dac id and float type
    df_wmo = pd.read_csv(
        f"{tmp_path}/{YOMAHA_URLS[3].split('/')[-1]}",
        sep=r"\s+",
        header=None,
        names=["id", "wmo_id", "dac_id", "float_type_id"],
        engine="python",
    )

    # mapping of Data Assembly Center (DAC) id and DAC name
    df_dac = pd.read_csv(
        f"{tmp_path}/{YOMAHA_URLS[1].split('/')[-1]}",
        sep=":",
        header=None,
        names=["dac_id", "dac"],
    )
    df_dac["dac"] = df_dac["dac"].str.strip()

    # mapping of float_type_id and float_type
    df_float = pd.read_csv(
        f"{tmp_path}/{YOMAHA_URLS[0].split('/')[-1]}",
        sep=":",
        header=None,
        skipfooter=4,
        names=["float_type_id", "float_type"],
        engine="python",
    )
    df_float.loc[df_float["float_type_id"] == 0, "float_type"] = "METOCEAN"

    df_wmo["dac_id"] = df_wmo["dac_id"].astype("int64")
    df_dac["dac_id"] = df_dac["dac_id"].astype("int64")

    # combine metadata, aligned to the sorted unique_id order
    df_metadata = (
        pd.merge(df_wmo, df_dac, on="dac_id", how="left")
        .merge(df_float, on="float_type_id", how="left")
        .loc[lambda x: np.isin(x["id"], unique_id)]
        .set_index("id")
        .reindex(unique_id)
        .reset_index()
    )

    # float64 variables to keep as float64
    double_vars = {
        "lat_d",
        "lon_d",
        "lat_s",
        "lon_s",
        "lat_lp",
        "lon_lp",
        "lat_fc",
        "lon_fc",
        "lat_lc",
        "lon_lc",
    }

    # time coordinate columns (stored as float, decoded by xarray via units attr)
    time_coord_names = ["time_d", "time_s", "time_lp", "time_fc", "time_lc"]

    # obs-level data columns (all except id and time coords)
    obs_data_names = [c for c in col_names if c not in ("id",) + tuple(time_coord_names)]

    def _cast(arr, name):
        if name in double_vars:
            return arr.to_numpy(dtype="float64")
        if arr.dtype == "float64":
            return arr.to_numpy(dtype="float32")
        return arr.to_numpy()

    coords = {
        "id": unique_id,
    }
    for tc in time_coord_names:
        coords[tc] = _cast(df[tc], tc)

    data = {c: _cast(df[c], c) for c in obs_data_names}

    attrs_variables = {
        "lon_d": {
            "long_name": "Longitude of the location where the deep velocity is calculated",
            "units": "degrees_east",
        },
        "lat_d": {
            "long_name": "Latitude of the location where the deep velocity is calculated",
            "units": "degrees_north",
        },
        "pres_d": {
            "long_name": "Reference parking pressure for this cycle",
            "units": "dbar",
        },
        "time_d": {
            "long_name": "Julian time (days) when deep velocity is estimated",
            "units": "days since 2000-01-01 00:00",
        },
        "ve_d": {
            "long_name": "Eastward component of the deep velocity",
            "units": "cm s-1",
        },
        "vn_d": {
            "long_name": "Northward component of the deep velocity",
            "units": "cm s-1",
        },
        "err_ve_d": {
            "long_name": "Error on the eastward component of the deep velocity",
            "units": "cm s-1",
        },
        "err_vn_d": {
            "long_name": "Error on the northward component of the deep velocity",
            "units": "cm s-1",
        },
        "lon_s": {
            "long_name": "Longitude of the location where the first surface velocity is calculated (over the first 6 h at surface)",
            "units": "degrees_east",
        },
        "lat_s": {
            "long_name": "Latitude of the location where the first surface velocity is calculated",
            "units": "degrees_north",
        },
        "time_s": {
            "long_name": "Julian time (days) when the first surface velocity is calculated",
            "units": "days since 2000-01-01 00:00",
        },
        "ve_s": {
            "long_name": "Eastward component of first surface velocity",
            "units": "cm s-1",
        },
        "vn_s": {
            "long_name": "Northward component of first surface velocity",
            "units": "cm s-1",
        },
        "err_ve_s": {
            "long_name": "Error on the eastward component of the first surface velocity",
            "units": "cm s-1",
        },
        "err_vn_s": {
            "long_name": "Error on the northward component of the first surface velocity",
            "units": "cm s-1",
        },
        "lon_lp": {
            "long_name": "Longitude of the last fix at the sea surface during the previous cycle",
            "units": "degrees_east",
        },
        "lat_lp": {
            "long_name": "Latitude of the last fix at the sea surface during the previous cycle",
            "units": "degrees_north",
        },
        "time_lp": {
            "long_name": "Julian time of the last fix at the sea surface during the previous cycle",
            "units": "days since 2000-01-01 00:00",
        },
        "lon_fc": {
            "long_name": "Longitude of the first fix at the sea surface during the current cycle",
            "units": "degrees_east",
        },
        "lat_fc": {
            "long_name": "Latitude of the first fix at the sea surface during the current cycle",
            "units": "degrees_north",
        },
        "time_fc": {
            "long_name": "Julian time of the first fix at the sea surface during the current cycle",
            "units": "days since 2000-01-01 00:00",
        },
        "lon_lc": {
            "long_name": "Longitude of the last fix at the sea surface during the current cycle",
            "units": "degrees_east",
        },
        "lat_lc": {
            "long_name": "Latitude of the last fix at the sea surface during the current cycle",
            "units": "degrees_north",
        },
        "time_lc": {
            "long_name": "Julian time of the last fix at the sea surface during the current cycle",
            "units": "days since 2000-01-01 00:00",
        },
        "surf_fix": {
            "long_name": "Number of surface fixes during the current cycle",
            "units": "-",
        },
        "id": {
            "long_name": "Float WMO number",
            "units": "-",
        },
        "cycle": {
            "long_name": "Cycle number",
            "units": "-",
        },
        "time_inv": {
            "long_name": "Time inversion/duplication flag",
            "description": "1 if at least one duplicate or inversion of time is found in the sequence containing last fix from the previous cycle and all fixes from the current cycle. Otherwise, 0.",
            "units": "-",
        },
        "wmo_id": {
            "long_name": "Float WMO number",
            "units": "-",
        },
        "dac_id": {
            "long_name": "Data Assembly Center (DAC) number",
            "description": "1: AOML (USA), 2: CORIOLIS (France), 3: JMA (Japan), 4: BODC (UK), 5: MEDS (Canada), 6: INCOIS (India), 7: KMA (Korea), 8: CSIRO (Australia), 9: CSIO (China)",
            "units": "-",
        },
        "float_type": {
            "long_name": "Float type",
            "description": "1: APEX, 2: SOLO, 3: PROVOR, 4: R1, 5: MARTEC, 6: PALACE, 7: NINJA, 8: NEMO, 9: ALACE, 0: METOCEAN",
            "units": "-",
        },
        "rowsize": {
            "long_name": "number of observations per trajectory",
            "sample_dimension": "obs",
            "units": "-",
        },
    }

    attrs_global = {
        "title": "YoMaHa'07: Velocity data assessed from trajectories of Argo floats at parking level and at the sea surface",
        "history": f"Dataset updated on {YOMAHA_VERSION}",
        "date_created": datetime.now().isoformat(),
        "publisher_name": "Asia-Pacific Data Research Center",
        "publisher_url": "http://apdrc.soest.hawaii.edu/index.php",
        "license": "freely available",
    }

    coord_dims = {"id": "traj"}
    for tc in time_coord_names:
        coord_dims[tc] = "obs"

    var_dims = {
        "rowsize": ["traj"],
        "wmo_id": ["traj"],
        "dac_id": ["traj"],
        "float_type": ["traj"],
    }
    for c in obs_data_names:
        var_dims[c] = ["obs"]

    return RaggedArray(
        coords=coords,
        metadata={
            "rowsize": rowsize.astype("int64"),
            "wmo_id": df_metadata["wmo_id"].to_numpy(),
            "dac_id": df_metadata["dac_id"].to_numpy(),
            "float_type": df_metadata["float_type_id"].to_numpy(),
        },
        data=data,
        attrs_global=attrs_global,
        attrs_variables=attrs_variables,
        name_dims={"traj": "rows", "obs": "obs"},
        coord_dims=coord_dims,
        var_dims=var_dims,
    )

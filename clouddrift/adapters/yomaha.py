"""
This module defines functions used to adapt the YoMaHa'07: Velocity data assessed
from trajectories of Argo floats at parking level and at the sea surface as
a ragged-arrays dataset.

The dataset is hosted at http://apdrc.soest.hawaii.edu/projects/yomaha/ and the user manual
is available at http://apdrc.soest.hawaii.edu/projects/yomaha/yomaha07/YoMaHa070612.pdf.

Example
-------
>>> from clouddrift.adapters import yomaha
>>> ds = yomaha.to_xarray()

Reference
---------
Lebedev, K. V., Yoshinari, H., Maximenko, N. A., & Hacker, P. W. (2007). Velocity data assessed from trajectories of Argo floats at parking level and at the sea surface. IPRC Technical Note, 4(2), 1-16.
"""

import gzip
import logging
import os
import sys
import tempfile
import warnings
from datetime import datetime
from io import BytesIO
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.adapters.utils import download_with_progress

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


def download(tmp_path: str):
    download_requests = [
        (url, f"{tmp_path}/{url.split('/')[-1]}", None) for url in YOMAHA_URLS[:-1]
    ]
    download_with_progress(download_requests)

    filename_gz = f"{tmp_path}/{YOMAHA_URLS[-1].split('/')[-1]}"
    filename = filename_gz.removesuffix(".gz")

    buffer = BytesIO()
    download_with_progress([(YOMAHA_URLS[-1], buffer, None)])

    decompressed_fp = os.path.join(tmp_path, filename)
    with open(decompressed_fp, "wb") as file:
        _logger.debug(
            f"decompressing {filename_gz} into {decompressed_fp}. Original Size: {sys.getsizeof(buffer)}"
        )
        buffer.seek(0)
        data = buffer.read()
        while data:
            file.write(gzip.decompress(data))
            data = buffer.read()
        _logger.debug(f"Decompressed size of {filename_gz}: {sys.getsizeof(file)}")
        buffer.close()


def to_xarray(tmp_path: Union[str, None] = None):
    if tmp_path is None:
        tmp_path = YOMAHA_TMP_PATH
        os.makedirs(tmp_path, exist_ok=True)

    # get or update required files
    download(tmp_path)

    # database last update
    with open(f"{tmp_path}/{YOMAHA_URLS[2].split('/')[-1]}") as f:
        YOMAHA_VERSION = f.read().strip()
        print(f"Last database update was: {YOMAHA_VERSION}")

    # parse with panda
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

    # open with pandas
    filename_gz = f"{tmp_path}/{YOMAHA_URLS[-1].split('/')[-1]}"
    filename = filename_gz.removesuffix(".gz")
    df = pd.read_csv(
        filename, names=col_names, sep=r"\s+", header=None, na_values=na_col
    )

    # convert to an Xarray Dataset
    ds = xr.Dataset.from_dataframe(df)

    unique_id, rowsize = np.unique(ds["id"], return_counts=True)

    # mapping of yomaha float id, wmo float id, daq id and float type
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
    # there is a note on METOCEAN * in float_types.txt but the
    # float id, wmo id, and float type do not match (?)
    # so we remove the * from the type
    df_float.loc[df_float["float_type_id"] == 0, "float_type"] = "METOCEAN"

    # combine metadata
    df_metadata = (
        pd.merge(df_wmo, df_dac, on="dac_id", how="left")
        .merge(df_float, on="float_type_id", how="left")
        .loc[lambda x: np.isin(x["id"], unique_id)]
    )

    ds = (
        ds.rename_dims({"index": "obs"})
        .assign({"id": ("traj", unique_id)})
        .assign({"rowsize": ("traj", rowsize)})
        .assign({"wmo_id": ("traj", df_metadata["wmo_id"])})
        .assign({"dac_id": ("traj", df_metadata["dac_id"])})
        .assign({"float_type": ("traj", df_metadata["float_type_id"])})
        .set_coords(["id", "time_d", "time_s", "time_lp", "time_lc", "time_lp"])
        .drop_vars(["index"])
    )

    # Cast double floats to singles
    double_vars = [
        "lat_d",
        "lon_d",
        "lat_s",
        "lon_s",
        "lat_lp",
        "lon_lp",
        "lat_lc",
        "lon_lc",
    ]
    for var in [v for v in ds.variables if v not in double_vars]:
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")

    # define attributes
    vars_attrs = {
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
    }

    # global attributes
    attrs = {
        "title": "YoMaHa'07: Velocity data assessed from trajectories of Argo floats at parking level and at the sea surface",
        "history": f"Dataset updated on {YOMAHA_VERSION}",
        "date_created": datetime.now().isoformat(),
        "publisher_name": "Asia-Pacific Data Research Center",
        "publisher_url": "http://apdrc.soest.hawaii.edu/index.php",
        "license": "freely available",
    }

    # set attributes
    for var in vars_attrs.keys():
        if var in ds.keys():
            ds[var].attrs = vars_attrs[var]
        else:
            warnings.warn(f"Variable {var} not found in upstream data; skipping.")
    ds.attrs = attrs

    return ds

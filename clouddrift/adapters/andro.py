"""
This module defines functions used to adapt the ANDRO dataset as
a ragged-array dataset. 

The dataset is hosted at https://www.seanoe.org/data/00360/47077/

Example
-------
>>> from clouddrift.adapters import andro
>>> ds = andro.to_xarray()

Reference
---------
Ollitrault Michel, Rannou Philippe, Brion Emilie, Cabanes Cecile, Piron Anne, Reverdin Gilles, Kolodziejczyk Nicolas (2022). ANDRO: An Argo-based deep displacement dataset. SEANOE. https://doi.org/10.17882/47077
"""

from clouddrift.adapters.yomaha import download_with_progress
import numpy as np
import os
import pandas as pd
import tempfile
import xarray as xr


# order of the URLs is important
ANDRO_URL = "https://www.seanoe.org/data/00360/47077/data/91950.dat"
ANDRO_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "andro")


def to_xarray(tmp_path: str = None):
    if tmp_path is None:
        tmp_path = ANDRO_TMP_PATH
        os.makedirs(tmp_path, exist_ok=True)

    # get or update dataset
    local_file = tmp_path + ANDRO_URL.split("/")[-1]
    download_with_progress(ANDRO_URL, local_file)

    # parse with panda
    col_names = [
        # depth
        "lon_d",
        "lat_d",
        "p_d",
        "temp_d",
        "s_d",
        "t_d",
        "u_d",
        "v_d",
        "eu_d",
        "ev_d",
        # first surface velocity
        "lon_s",
        "lat_s",
        "t_s",
        "u_s",
        "v_s",
        "eu_s",
        "ev_s",
        # last surface velocity
        "lon_ls",
        "lat_ls",
        "t_ls",
        "u_ls",
        "v_ls",
        "eu_ls",
        "ev_ls",
        # last fix previous cycle
        "lon_lp",
        "lat_lp",
        "t_lp",
        # first fix current cycle
        "lon_fc",
        "lat_fc",
        "t_fc",
        # last fix current cycle
        "lon_lc",
        "lat_lc",
        "t_lc",
        "s_fix",
        "id",
        "cycle",
        "t_inv",
    ]

    na_col = [
        -999.9999,
        -99.9999,
        -999.9,
        -99.999,
        -99.999,
        -9999.999,
        -999.9,
        -999.9,
        -999.9,
        -999.9,
        -999.9999,
        -99.999,
        -9999.999,
        -999.99,
        -999.99,
        -999.99,
        -999.99,
        -999.9999,
        -99.9999,
        -9999.999,
        -999.99,
        -999.99,
        -999.99,
        -999.99,
        -999.9999,
        -99.9999,
        -9999.999,
        -999.9999,
        -99.9999,
        -9999.999,
        -999.9999,
        -99.9999,
        -9999.999,
        np.nan,
        np.nan,
        np.nan,
        -99,
    ]

    # open with pandas
    local_file = ANDRO_URL.split("/")[-1]
    df = pd.read_csv(
        local_file, names=col_names, sep="\s+", header=None, na_values=na_col
    )

    # convert to an Xarray Dataset
    ds = xr.Dataset.from_dataframe(df)
    ds = ds.rename_dims({"index": "obs"})

    for t in ["t_s", "t_d", "t_lp", "t_fc", "t_lc"]:
        ds[t].values = pd.to_datetime(ds[t], origin="2000-01-01 00:00", unit="D").values

    unique_id, rowsize = np.unique(ds["id"], return_counts=True)

    ds["id"] = (["traj"], unique_id)
    ds["rowsize"] = (["traj"], rowsize)

    ds = ds.set_coords(["id", "t_d", "t_s", "t_lp", "t_lc", "t_lp"])
    ds = ds.drop_vars("index")

    return ds

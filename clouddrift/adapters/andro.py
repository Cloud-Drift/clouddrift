"""
This module defines functions used to adapt the ANDRO: An Argo-based
deep displacement dataset as a ragged-arrays dataset.

The dataset is hosted at https://www.seanoe.org/data/00360/47077/ and the user manual
is available at https://archimer.ifremer.fr/doc/00360/47126/.

Example
-------
>>> from clouddrift.adapters import andro
>>> ds = andro.to_xarray()

Reference
---------
Ollitrault Michel, Rannou Philippe, Brion Emilie, Cabanes Cecile, Piron Anne, Reverdin Gilles,
Kolodziejczyk Nicolas (2022). ANDRO: An Argo-based deep displacement dataset.
SEANOE. https://doi.org/10.17882/47077
"""

import os
import tempfile
import warnings
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.adapters.utils import download_with_progress

# order of the URLs is important
ANDRO_URL = "https://www.seanoe.org/data/00360/47077/data/91950.dat"
ANDRO_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "andro")
ANDRO_VERSION = "2022-03-04"


def to_xarray(tmp_path: Union[str, None] = None):
    if tmp_path is None:
        tmp_path = ANDRO_TMP_PATH
        os.makedirs(tmp_path, exist_ok=True)

    # get or update dataset
    local_file = f"{tmp_path}/{ANDRO_URL.split('/')[-1]}"
    download_with_progress([(ANDRO_URL, local_file, None)])

    # parse with panda
    col_names = [
        # depth
        "lon_d",
        "lat_d",
        "pres_d",
        "temp_d",
        "sal_d",
        "time_d",
        "ve_d",
        "vn_d",
        "err_ve_d",
        "err_vn_d",
        # first surface velocity
        "lon_s",
        "lat_s",
        "time_s",
        "ve_s",
        "vn_s",
        "err_ve_s",
        "err_vn_s",
        # last surface velocity
        "lon_ls",
        "lat_ls",
        "time_ls",
        "ve_ls",
        "vn_ls",
        "err_ve_ls",
        "err_vn_ls",
        # last fix previous cycle
        "lon_lp",
        "lat_lp",
        "time_lp",
        # first fix current cycle
        "lon_fc",
        "lat_fc",
        "time_fc",
        # last fix current cycle
        "lon_lc",
        "lat_lc",
        "time_lc",
        "surf_fix",
        "id",
        "cycle",
        "profile_id",
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
    df = pd.read_csv(
        local_file,
        names=col_names,
        sep=r"\s+",
        header=None,
        na_values=na_col,  # type: ignore
    )

    # convert to an Xarray Dataset
    ds = xr.Dataset.from_dataframe(df)

    unique_id, rowsize = np.unique(ds["id"], return_counts=True)

    ds = (
        ds.rename_dims({"index": "obs"})
        .assign({"id": ("traj", unique_id)})
        .assign({"rowsize": ("traj", rowsize)})
        .set_coords(["id", "time_d", "time_s", "time_lp", "time_lc", "time_lp"])
        .drop_vars(["index"])
    )

    # Cast double floats to singles
    double_vars = [
        "lat_d",
        "lon_d",
        "lat_s",
        "lon_s",
        "lat_ls",
        "lon_ls",
        "lat_lp",
        "lon_lp",
        "lat_fc",
        "lon_fc",
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
        "temp_d": {
            "long_name": "Parking temperature (°C) for this cycle",
            "units": "degree_C",
        },
        "sal_d": {
            "long_name": "Parking salinity for this cycle",
            "units": "psu",
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
        "lon_ls": {
            "long_name": "Longitude of the location where the last surface velocity is calculated (over the last 6 h at surface)",
            "units": "degrees_east",
        },
        "lat_ls": {
            "long_name": "Latitude of the location where the last surface velocity is calculated",
            "units": "degrees_north",
        },
        "time_ls": {
            "long_name": "Julian time (days) when the last surface velocity is calculated",
            "units": "days since 2000-01-01 00:00",
        },
        "ve_ls": {
            "long_name": "Eastward component of last surface velocity (cm s-1)",
            "units": "cm s-1",
        },
        "vn_ls": {
            "long_name": "Northward component of last surface velocity (cm s-1)",
            "units": "cm s-1",
        },
        "err_ve_ls": {
            "long_name": "Error on the eastward component of the last surface velocity",
            "units": "cm s-1",
        },
        "err_vn_ls": {
            "long_name": "Error on the northward component of the last surface velocity",
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
        "profile_id": {
            "long_name": "Profile number as given in the NetCDF prof file",
            "units": "-",
        },
    }

    # global attributes
    attrs = {
        "title": "ANDRO: An Argo-based deep displacement dataset",
        "history": f"Dataset updated on {ANDRO_VERSION}",
        "date_created": datetime.now().isoformat(),
        "publisher_name": "SEANOE (SEA scieNtific Open data Edition)",
        "publisher_url": "https://www.seanoe.org/data/00360/47077/",
        "license": "Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/)",
    }

    # set attributes
    for var in vars_attrs.keys():
        if var in ds.keys():
            ds[var].attrs = vars_attrs[var]
        else:
            warnings.warn(f"Variable {var} not found in upstream data; skipping.")
    ds.attrs = attrs

    return ds

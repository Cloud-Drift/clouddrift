"""
This module defines functions used to adapt the ANDRO: An Argo-based
deep displacement dataset as a ragged-arrays dataset.

The dataset is hosted at https://www.seanoe.org/data/00360/47077/ and the user manual
is available at https://archimer.ifremer.fr/doc/00360/47126/.

Example
-------
>>> from clouddrift.adapters import andro
>>> ra = andro.to_raggedarray()
>>> ds = ra.to_xarray()

Reference
---------
Ollitrault Michel, Rannou Philippe, Brion Emilie, Cabanes Cecile, Piron Anne, Reverdin Gilles,
Kolodziejczyk Nicolas (2022). ANDRO: An Argo-based deep displacement dataset.
SEANOE. https://doi.org/10.17882/47077
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

# order of the URLs is important
ANDRO_URL = "https://www.seanoe.org/data/00360/47077/data/127690.zip"
ANDRO_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "andro")
ANDRO_VERSION = "2026-04"


def to_raggedarray(
    tmp_path: str | None = None, skip_download: bool = False
) -> RaggedArray:
    """Return the ANDRO dataset as a RaggedArray instance.

    Parameters
    ----------
    tmp_path : str, optional
        Path where the dataset file is cached. Defaults to a platform-specific temporary directory.
    skip_download : bool, optional
        If True, skip re-downloading the dataset file if it already exists in ``tmp_path``. Default is False.
    """
    if tmp_path is None:
        tmp_path = ANDRO_TMP_PATH
    os.makedirs(tmp_path, exist_ok=True)

    # get or update dataset
    local_file = f"{tmp_path}/{ANDRO_URL.split('/')[-1]}"
    download_with_progress([(ANDRO_URL, local_file)], skip_download=skip_download)

    # parse with pandas
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

    ids = df["id"].to_numpy()
    traj, rowsize = np.unique(ids, return_counts=True)

    attrs_global = {
        "title": "ANDRO: An Argo-based deep displacement dataset (Quality controlled data)",
        "history": f"Dataset updated on {ANDRO_VERSION}",
        "date_created": datetime.now().isoformat(),
        "publisher_name": "SEANOE (SEA scieNtific Open data Edition)",
        "publisher_url": "https://www.seanoe.org/data/00360/47077/",
        "license": "Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/)",
    }

    vars_attrs = {
        "rowsize": {
            "long_name": "Number of observations for each trajectory",
            "sample_dimension": "obs",
            "units": "-",
        },
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

    attrs_variables = {
        k: v for k, v in vars_attrs.items() if k in df.columns or k in ["id", "rowsize"]
    }

    # Preserve historical dtype behavior: keep lon/lat in float64 and downcast
    # other float64 variables to float32 to limit memory usage.
    double_vars = {
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
    }

    # Only keep columns that are in the DataFrame
    data_vars = {}
    for k in [name for name in df.columns if name != "id"]:
        values = df[k].to_numpy()
        if values.dtype == np.float64 and k not in double_vars:
            values = values.astype(np.float32)
        data_vars[k] = values

    # Extract time coordinates from data_vars
    time_coords = {}
    time_coord_names = ["time_d", "time_s", "time_lp", "time_lc", "time_fc"]
    for name in time_coord_names:
        if name in data_vars:
            time_coords[name] = data_vars.pop(name)

    return RaggedArray(
        coords={
            "id": traj,
            **time_coords,
        },
        metadata={
            "rowsize": rowsize.astype("int64"),
        },
        data=data_vars,
        attrs_global=attrs_global,
        attrs_variables=attrs_variables,
        name_dims={"traj": "rows", "obs": "obs"},
        coord_dims={
            "id": "traj",
            **{name: "obs" for name in time_coords.keys()},
        },
        var_dims={
            "rowsize": ["traj"],
            **{k: ["obs"] for k in data_vars.keys()},
        },
    )

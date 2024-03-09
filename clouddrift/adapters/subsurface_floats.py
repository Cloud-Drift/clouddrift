"""
This module defines functions to adapt as a ragged-array dataset a collection of data
from 2193 trajectories of SOFAR, APEX, and RAFOS subsurface floats from 52 experiments
across the world between 1989 and 2015.

The dataset is hosted at https://www.aoml.noaa.gov/phod/float_traj/index.php

Example
-------
>>> from clouddrift.adapters import subsurface_floats
>>> ds = subsurface_floats.to_xarray()
"""

import os
import tempfile
import warnings
from datetime import datetime
from typing import Hashable, List, Union

import numpy as np
import pandas as pd
import scipy.io  # type: ignore
import xarray as xr

from clouddrift.adapters.utils import download_with_progress

SUBSURFACE_FLOATS_DATA_URL = (
    "https://www.aoml.noaa.gov/phod/float_traj/files/allFloats_12122017.mat"
)
SUBSURFACE_FLOATS_VERSION = "December 2017 (version 2)"
SUBSURFACE_FLOATS_TMP_PATH = os.path.join(
    tempfile.gettempdir(), "clouddrift", "subsurface_floats"
)


def download(file: str):
    download_with_progress([(SUBSURFACE_FLOATS_DATA_URL, file, None)])


def to_xarray(
    tmp_path: Union[str, None] = None,
):
    if tmp_path is None:
        tmp_path = SUBSURFACE_FLOATS_TMP_PATH
        os.makedirs(tmp_path, exist_ok=True)

    local_file = f"{tmp_path}/{SUBSURFACE_FLOATS_DATA_URL.split('/')[-1]}"
    download(local_file)
    source_data = scipy.io.loadmat(local_file)

    # metadata
    meta_variables: List[Hashable] = [
        "expList",
        "expName",
        "expOrg",
        "expPI",
        "fltType",
        "indexExp",
        "indexFlt",
    ]

    metadata = {}
    for var in meta_variables:
        metadata[var] = np.array([v.flatten()[0] for v in source_data[var].flatten()])

    # bring the expList to the "traj" dimension
    _, float_per_exp = np.unique(metadata["indexExp"], return_counts=True)
    metadata["expList"] = np.repeat(metadata["expList"], float_per_exp)

    # data
    data_variables = ["dtnum", "lon", "lat", "p", "t", "u", "v"]
    data = {}
    for var in data_variables:
        data[var] = np.concatenate([v.flatten() for v in source_data[var].flatten()])

    # create rowsize variable
    rowsize = np.array([len(v) for v in source_data["dtnum"].flatten()])
    assert np.sum(rowsize) == len(data["dtnum"])

    # Unix epoch start (1970-01-01)
    origin_datenum = 719529

    ds = xr.Dataset(
        {
            "expList": (["traj"], metadata["expList"]),
            "expName": (["traj"], metadata["expName"]),
            "expOrg": (["traj"], metadata["expOrg"]),
            "expPI": (["traj"], metadata["expPI"]),
            "indexExp": (["traj"], metadata["indexExp"]),
            "fltType": (["traj"], metadata["fltType"]),
            "id": (["traj"], metadata["indexFlt"]),
            "rowsize": (["traj"], rowsize),
            "time": (
                ["obs"],
                pd.to_datetime(data["dtnum"] - origin_datenum, unit="D"),
            ),
            "lon": (["obs"], data["lon"]),
            "lat": (["obs"], data["lat"]),
            "pres": (["obs"], data["p"]),
            "temp": (["obs"], data["t"]),
            "ve": (["obs"], data["u"]),
            "vn": (["obs"], data["v"]),
        }
    )

    # Cast double floats to singles
    double_vars = ["lat", "lon"]
    for var in [v for v in ds.variables if v not in double_vars]:
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")

    # define attributes
    vars_attrs = {
        "expList": {
            "long_name": "Experiment list",
            "units": "-",
        },
        "expName": {
            "long_name": "Experiment name",
            "units": "-",
        },
        "expOrg": {
            "long_name": "Experiment organization",
            "units": "-",
        },
        "expPI": {
            "long_name": "Experiment principal investigator",
            "units": "-",
        },
        "indexExp": {
            "long_name": "Experiment index number",
            "units": "-",
            "comment": "The index matches the float with its experiment metadata",
        },
        "fltType": {
            "long_name": "Float type",
            "units": "-",
        },
        "id": {"long_name": "Float ID", "units": "-"},
        "lon": {
            "long_name": "Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        },
        "lat": {
            "long_name": "Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        },
        "rowsize": {
            "long_name": "Number of observations per trajectory",
            "sample_dimension": "obs",
            "units": "-",
        },
        "pres": {
            "long_name": "Pressure",
            "standard_name": "sea_water_pressure",
            "units": "dbar",
        },
        "temp": {
            "long_name": "Temperature",
            "standard_name": "sea_water_temperature",
            "units": "degree_C",
        },
        "ve": {
            "long_name": "Eastward velocity",
            "standard_name": "eastward_sea_water_velocity",
            "units": "m s-1",
        },
        "vn": {
            "long_name": "Northward velocity",
            "standard_name": "northward_sea_water_velocity",
            "units": "m s-1",
        },
    }

    # global attributes
    attrs = {
        "title": "Subsurface float trajectories dataset",
        "history": SUBSURFACE_FLOATS_VERSION,
        "date_created": datetime.now().isoformat(),
        "publisher_name": "WOCE Subsurface Float Data Assembly Center and NOAA AOML",
        "publisher_url": "https://www.aoml.noaa.gov/phod/float_traj/data.php",
        "license": "freely available",
        "acknowledgement": "Maintained by Andree Ramsey and Heather Furey from the Woods Hole Oceanographic Institution",
    }

    # set attributes
    for var in vars_attrs.keys():
        if var in ds.keys():
            ds[var].attrs = vars_attrs[var]
        else:
            warnings.warn(f"Variable {var} not found in upstream data; skipping.")
    ds.attrs = attrs

    # set coordinates
    ds = ds.set_coords(["time", "id"])

    return ds

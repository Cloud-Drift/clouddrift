"""
This module defines functions used to adapt the subsurface float trajectories as
a ragged-array dataset.

The dataset is hosted at https://www.aoml.noaa.gov/phod/float_traj/index.php

Example
-------
>>> from clouddrift.adapters import subsurface
>>> ds = subsurface.to_xarray()
"""

import scipy.io
import urllib.request
import os
import tempfile
import xarray as xr
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

SUBSURFACE_FLOAT_DATA_URL = (
    "https://www.aoml.noaa.gov/phod/float_traj/files/allFloats_12122017.mat"
)
SUBSURFACE_FLOAT_VERSION = f"December 2017 (version 2)"
SUBSURFACE_FLOAT_TMP_PATH = os.path.join(
    tempfile.gettempdir(), "clouddrift", "subsurface_float"
)


def download(file: str):
    print(
        f"Downloading Subsurface float trajectories from {SUBSURFACE_FLOAT_DATA_URL} to {file}..."
    )
    if not os.path.isfile(file):
        urllib.request.urlretrieve(SUBSURFACE_FLOAT_DATA_URL, file)
    else:
        warnings.warn(f"{file} already exists; skip download.")


def to_xarray(
    tmp_path: str = None,
):
    if tmp_path is None:
        tmp_path = SUBSURFACE_FLOAT_TMP_PATH
        os.makedirs(tmp_path, exist_ok=True)

    local_file = f"{tmp_path}/{SUBSURFACE_FLOAT_DATA_URL.split('/')[-1]}"
    download(local_file)
    source_data = scipy.io.loadmat(local_file)

    # metadata
    meta_variables = [
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

    # data
    data_variables = ["dtnum", "lon", "lat", "p", "t", "u", "v"]
    data = {}
    for var in data_variables:
        data[var] = np.concatenate([v.flatten() for v in source_data[var].flatten()])

    # create rowsize variable
    rowsize = np.array([len(v) for v in source_data["dtnum"].flatten()])
    assert np.sum(rowsize) == len(data["dtnum"])

    # some metadata are repeated for each float
    # use those indices to retrieve one value per experiment
    _, indices_exp = np.unique(metadata["indexExp"], return_index=True)

    origin_datenum = 719529  # Unix epoch start (1970-01-01)

    ds = xr.Dataset(
        {
            "expList": (["exp"], metadata["expList"]),
            "expName": (["exp"], metadata["expName"][indices_exp]),
            "expOrg": (["exp"], metadata["expOrg"][indices_exp]),
            "expPI": (["exp"], metadata["expPI"][indices_exp]),
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
            "p": (["obs"], data["p"]),
            "t": (["obs"], data["t"]),
            "u": (["obs"], data["u"]),
            "v": (["obs"], data["v"]),
        }
    )

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
        "lon": {"long_name": "Longitude", "units": "degrees_east"},
        "lat": {"long_name": "Latitude", "units": "degrees_north"},
        "rowsize": {
            "long_name": "Number of observations per trajectory",
            "sample_dimension": "obs",
            "units": "-",
        },
        "u": {"long_name": "Eastward velocity", "units": "m/s"},
        "v": {"long_name": "Northward velocity", "units": "m/s"},
        "t": {"long_name": "Temperature", "units": "Celsius"},
        "p": {"long_name": "Pressure", "units": "Millibar"},
    }

    # global attributes
    attrs = {
        "title": "Subsurface float trajectories dataset",
        "history": SUBSURFACE_FLOAT_VERSION,
        "date_created": datetime.now().isoformat(),
        "publisher_name": "WOCE Subsurface Float Data Assembly Center and NOAA AOML",
        "publisher_url": "https://www.aoml.noaa.gov/phod/float_traj/data.php",
        "licence": "freely available",
        "acknowledgement": f"Maintained by Andree Ramsey and Heather Furey from the Woods Hole Oceanographic Institution",
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

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

SUBSURFACE_FLOAT_DATA_URL = (
    "https://www.aoml.noaa.gov/phod/float_traj/files/allFloats_12122017.mat"
)
SUBSURFACE_FLOAT_TMP_PATH = os.path.join(
    tempfile.gettempdir(), "clouddrift", "subsurface_float"
)


def download(tmp_path):
    print(
        f"Downloading Subsurface float trajectories from {SUBSURFACE_FLOAT_DATA_URL} to {tmp_path}..."
    )
    os.makedirs(tmp_path, exist_ok=True)
    urllib.request.urlretrieve(SUBSURFACE_FLOAT_DATA_URL, tmp_path)
    return


def to_xarray(
    tmp_path: str = None,
):
    if tmp_path is None:
        tmp_path = SUBSURFACE_FLOAT_TMP_PATH
    local_file = f"{tmp_path}/{url.split('/')[-1]}"

    if not local_file:
        download(tmp_path)
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
        metadata[var] = source_data[var].flatten()

    # data
    data_variables = ["dtnum", "lon", "lat", "p", "t", "u", "v"]
    data = {}
    for var in data_variables:
        data[var] = np.concatenate([v.flatten() for v in source_data[var].flatten()])

    # some metadata are repeated for each float
    # use those indices to retrieve one value per experiment
    _, indices_exp = np.unique(metadata["indexExp"], return_index=True)

    ds = xr.Dataset(
        {
            "experiment_list": (["exp"], metadata["expList"]),
            "experiment_name": (["exp"], metadata["expName"][indices_exp]),
            "experiment_org": (["exp"], metadata["expOrg"][indices_exp]),
            "experiment_pi": (["exp"], metadata["expPI"][indices_exp]),
            "index_exp": (["traj"], metadata["indexExp"]),
            "float_type": (["traj"], metadata["fltType"]),
            "id": (["traj"], metadata["indexFlt"]),
            "date": (["obs"], data["dtnum"]),
            "date": (["obs"], data["dtnum"]),
            "date": (["obs"], data["dtnum"]),
            "date": (["obs"], data["dtnum"]),
            "lon": (["obs"], data["lon"]),
            "lat": (["obs"], data["lat"]),
            "pressure": (["obs"], data["p"]),
            "temperature": (["obs"], data["t"]),
            "ve": (["obs"], data["u"]),
            "vn": (["obs"], data["v"]),
        }
    )
    return ds

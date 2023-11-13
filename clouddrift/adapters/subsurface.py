"""
This module defines functions used to adapt the subsurface float trajectories as
a ragged-array dataset.

The dataset is hosted at https://www.aoml.noaa.gov/phod/float_traj/index.php

Example
-------
>>> from clouddrift.adapters import subsurface
>>> ds = subsurface.to_xarray()
"""

from clouddrift.adapters.gdp import cut_str
import scipy.io
import urllib.request
import os
import tempfile
import xarray as xr
import numpy as np
import warnings

SUBSURFACE_FLOAT_DATA_URL = (
    "https://www.aoml.noaa.gov/phod/float_traj/files/allFloats_12122017.mat"
)
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
        metadata[var] = source_data[var].flatten()

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

    # todo 2 to fix netcdf export maybe...
    # "experiment_list": (["exp"], cut_str(metadata["expList"], 20)),
    # "experiment_name": (["exp"], cut_str(metadata["expName"][indices_exp], 20)),
    # "experiment_org": (["exp"], cut_str(metadata["expOrg"][indices_exp], 20)),
    # "experiment_pi": (["exp"], cut_str(metadata["expPI"][indices_exp]), 20),

    ds = xr.Dataset(
        {
            "experiment_list": (["exp"], metadata["expList"]),
            "experiment_name": (["exp"], metadata["expName"][indices_exp]),
            "experiment_org": (["exp"], metadata["expOrg"][indices_exp]),
            "experiment_pi": (["exp"], metadata["expPI"][indices_exp]),
            "index_exp": (["traj"], metadata["indexExp"]),
            "float_type": (["traj"], metadata["fltType"]),
            "id": (["traj"], metadata["indexFlt"]),
            "rowsize": (["traj"], rowsize),
            "datenum": (["obs"], data["dtnum"]),
            "ids": (["obs"], np.repeat(metadata["indexFlt"], rowsize)),
            "lon": (["obs"], data["lon"]),
            "lat": (["obs"], data["lat"]),
            "pressure": (["obs"], data["p"]),
            "temperature": (["obs"], data["t"]),
            "ve": (["obs"], data["u"]),
            "vn": (["obs"], data["v"]),
        }
    )

    # set coordinates
    ds = ds.set_coords(["datenum", "ids"])

    return ds

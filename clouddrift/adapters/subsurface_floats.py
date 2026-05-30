"""
This module defines functions to adapt as a ragged-array dataset a collection of data
from 2193 trajectories of SOFAR, APEX, and RAFOS subsurface floats from 52 experiments
across the world between 1989 and 2015.

The dataset is hosted at https://www.aoml.noaa.gov/phod/float_traj/index.php

Example
-------
>>> from clouddrift.adapters import subsurface_floats
>>> ra = subsurface_floats.to_raggedarray()
"""

import os
import tempfile
from collections.abc import Hashable
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io  # type: ignore

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

SUBSURFACE_FLOATS_DATA_URL = (
    "https://www.aoml.noaa.gov/phod/float_traj/files/allFloats_12122017.mat"
)
SUBSURFACE_FLOATS_VERSION = "December 2017 (version 2)"
SUBSURFACE_FLOATS_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "subsurface_floats")


def download(file: str, skip_download: bool = False):
    download_with_progress([(SUBSURFACE_FLOATS_DATA_URL, file)], skip_download=skip_download)


def to_raggedarray(
    tmp_path: str | None = None,
    skip_download: bool = False,
) -> RaggedArray:
    """Convert the subsurface floats dataset to a RaggedArray instance.

    Parameters
    ----------
    tmp_path : str, optional
        Path where the dataset file is cached. Defaults to a platform-specific
        temporary directory.
    skip_download : bool, optional
        If True, skip re-downloading the dataset file if it already exists in
        ``tmp_path``. Default is False.

    Returns
    -------
    RaggedArray
        Subsurface float trajectories as a ragged array.
    """
    if tmp_path is None:
        tmp_path = SUBSURFACE_FLOATS_TMP_PATH
    os.makedirs(tmp_path, exist_ok=True)

    local_file = f"{tmp_path}/{SUBSURFACE_FLOATS_DATA_URL.split('/')[-1]}"
    download(local_file, skip_download=skip_download)
    if os.path.getsize(local_file) == 0:
        raise ConnectionError(
            f"Got empty response from subsurface floats server (url={SUBSURFACE_FLOATS_DATA_URL})"
        )
    source_data = scipy.io.loadmat(local_file)

    # metadata
    meta_variables: list[Hashable] = [
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
        arrs = _to_dense_flatten(source_data[str(var)])
        metadata[var] = np.array([_flatten_array(v)[0] for v in arrs])

    # bring the expList to the "traj" dimension
    _, float_per_exp = np.unique(metadata["indexExp"], return_counts=True)
    metadata["expList"] = np.repeat(metadata["expList"], float_per_exp)

    # data
    data_variables = ["dtnum", "lon", "lat", "p", "t", "u", "v"]
    raw_data = {}
    for var in data_variables:
        arrs = _to_dense_flatten(source_data[str(var)])
        raw_data[var] = np.concatenate([_flatten_array(v) for v in arrs])

    # create rowsize variable
    arrs = _to_dense_flatten(source_data["dtnum"])
    rowsize = np.array([len(_flatten_array(v)) for v in arrs])

    # Unix epoch start (1970-01-01)
    origin_datenum = 719529
    time = pd.to_datetime(raw_data["dtnum"] - origin_datenum, unit="D").to_numpy(
        dtype="datetime64[ns]"
    )

    attrs_variables = {
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
        "time": {"long_name": "time"},
        "rowsize": {
            "long_name": "Number of observations per trajectory",
            "sample_dimension": "obs",
            "units": "-",
        },
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

    attrs_global = {
        "title": "Subsurface float trajectories dataset",
        "history": SUBSURFACE_FLOATS_VERSION,
        "date_created": datetime.now().isoformat(),
        "publisher_name": "WOCE Subsurface Float Data Assembly Center and NOAA AOML",
        "publisher_url": "https://www.aoml.noaa.gov/phod/float_traj/data.php",
        "license": "freely available",
        "acknowledgement": "Maintained by Andree Ramsey and Heather Furey from the Woods Hole Oceanographic Institution",
    }

    return RaggedArray(
        coords={
            "id": metadata["indexFlt"],
            "time": time,
        },
        metadata={
            "rowsize": rowsize.astype("int64"),
            "expList": metadata["expList"],
            "expName": metadata["expName"],
            "expOrg": metadata["expOrg"],
            "expPI": metadata["expPI"],
            "indexExp": metadata["indexExp"],
            "fltType": metadata["fltType"],
        },
        data={
            "lon": raw_data["lon"],
            "lat": raw_data["lat"],
            "pres": raw_data["p"].astype("float32"),
            "temp": raw_data["t"].astype("float32"),
            "ve": raw_data["u"].astype("float32"),
            "vn": raw_data["v"].astype("float32"),
        },
        attrs_global=attrs_global,
        attrs_variables=attrs_variables,
        name_dims={"traj": "rows", "obs": "obs"},
        coord_dims={"id": "traj", "time": "obs"},
        var_dims={
            "rowsize": ["traj"],
            "expList": ["traj"],
            "expName": ["traj"],
            "expOrg": ["traj"],
            "expPI": ["traj"],
            "indexExp": ["traj"],
            "fltType": ["traj"],
            "lon": ["obs"],
            "lat": ["obs"],
            "pres": ["obs"],
            "temp": ["obs"],
            "ve": ["obs"],
            "vn": ["obs"],
        },
    )


def _flatten_array(arr):
    # Convert sparse to dense if needed, then flatten
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.array(arr).flatten()


def _to_dense_flatten(arr):
    """Convert a possibly sparse array to dense and flatten it."""
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.array(arr).flatten()

import numpy as np
import xarray as xr
import urllib.request
import os
from os.path import isfile, join, exists

folder = "../data/raw/parcels/"
file = "example.nc"
os.makedirs(folder, exist_ok=exists(folder))  # create raw data folder

# download and parse the file
if not isfile(join(folder, file)):
    url = "https://zenodo.org/record/6310460/files/global-marine-litter-2021.nc"
    print(f"Downloading ~1.1GB from {url}.")
    req = urllib.request.urlretrieve(url, join(folder, file))
    print(f"Dataset saved at {join(folder, file)}")
else:
    print(f"Dataset already at {join(folder, file)}.")

# load the two-dimensional Dataset
ds = xr.open_dataset(join(folder, file))
finite_values = np.isfinite(ds["lon"])
idx_finite = np.where(finite_values)
rowsize_ = np.bincount(idx_finite[0]).astype("int32")


def rowsize(index: int) -> int:
    return rowsize_[index]


def preprocess(index: int) -> xr.Dataset:
    """
    Extract the Lagrangian data for one trajectory from a Ocean Parcels dataset

    :param index: trajectory identification number
    :return: xr.Dataset containing the data and attributes
    """
    # subset the main file
    ds_subset = ds.isel(traj=[index], obs=finite_values[index])

    # add variables
    ds_subset["ID"] = (("traj"), [index])
    ds_subset["rowsize"] = (("traj"), [ds_subset.dims["obs"]])
    ds_subset["ids"] = (
        ("obs"),
        np.ones(ds_subset.dims["obs"], dtype="int") * index,
    )

    return ds_subset

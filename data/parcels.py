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
    # TODO: select a file available online
    # req = urllib.request.urlretrieve(griidc_url, join(folder, file))
    # print(f"Dataset {file} downloaded to '{folder}'.")
    print(f"Dataset {file} not found in '{folder}'.")
else:
    print(f"Dataset {file} already in '{folder}'.")


# load the two-dimensional Dataset
ds = xr.open_dataset(join(folder, file))


def preprocess(index: int) -> xr.Dataset:
    """
    Extract the Lagrangian data for one trajectory from a Ocean Parcels dataset
    :param index: trajectory identification number
    :return: xr.Dataset containing the data and attributes
    """
    # subset the main file
    finite_values = np.where(np.isfinite(ds.trajectory[index, :]))[0]
    ds_subset = ds.isel(traj=[index], obs=finite_values)

    # add variables
    trajectory_id = int(ds_subset.trajectory[0, 0].data)
    ds_subset["ID"] = (("traj"), [trajectory_id])
    ds_subset["rowsize"] = (("traj"), [ds_subset.dims["obs"]])
    ds_subset["ids"] = (
        ("obs"),
        np.ones(ds_subset.dims["obs"], dtype="int") * trajectory_id,
    )

    return ds_subset

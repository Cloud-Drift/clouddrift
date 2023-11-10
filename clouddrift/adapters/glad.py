"""
This module defines functions used to adapt the Grand LAgrangian Deployment
(GLAD) dataset as a ragged-array Xarray Dataset.

The dataset and its description are hosted at https://doi.org/10.7266/N7VD6WC8.

Example
-------
>>> from clouddrift.adapters import glad
>>> ds = glad.to_xarray()
"""
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr


def get_dataframe() -> pd.DataFrame:
    """Get the GLAD dataset as a pandas DataFrame."""
    url = "https://data.gulfresearchinitiative.org/pelagos-symfony/api/file/download/169841"
    # GRIIDC server doesn't provide Content-Length header, so we'll hardcode
    # the expected data length here.
    file_size = 155330876
    r = requests.get(url, stream=True)
    progress_bar = tqdm.tqdm(total=file_size, unit="iB", unit_scale=True)
    buf = StringIO()
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            buf.write(chunk.decode("utf-8"))
            progress_bar.update(len(chunk))
    buf.seek(0)
    progress_bar.close()
    column_names = [
        "id",
        "date",
        "time",
        "latitude",
        "longitude",
        "position_error",
        "u",
        "v",
        "velocity_error",
    ]
    df = pd.read_csv(buf, delim_whitespace=True, skiprows=5, names=column_names)
    df["obs"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.drop(["date", "time"], axis=1, inplace=True)
    return df


def to_xarray() -> xr.Dataset:
    """Return the GLAD data as an ragged-array Xarray Dataset."""
    df = get_dataframe()
    ds = df.to_xarray()

    traj, rowsize = np.unique(ds.id, return_counts=True)

    # Make the dataset compatible with clouddrift functions.
    ds = (
        ds.swap_dims({"index": "obs"})
        .assign_coords(obs=ds.obs, traj=traj)
        .assign({"rowsize": ("traj", rowsize)})
        .drop_vars(["id", "index"])
    )

    # Cast double floats to singles
    for var in ds.variables:
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")

    # Set variable attributes
    ds["longitude"].attrs = {
        "long_name": "longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
    }

    ds["latitude"].attrs = {
        "long_name": "latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
    }

    ds["position_error"].attrs = {
        "long_name": "position_error",
        "units": "m",
    }

    ds["u"].attrs = {
        "long_name": "eastward_sea_water_velocity",
        "standard_name": "eastward_sea_water_velocity",
        "units": "m s-1",
    }

    ds["v"].attrs = {
        "long_name": "northward_sea_water_velocity",
        "standard_name": "northward_sea_water_velocity",
        "units": "m s-1",
    }

    ds["velocity_error"].attrs = {
        "long_name": "velocity_error",
        "units": "m s-1",
    }

    return ds

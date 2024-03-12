"""
This module defines functions used to adapt the Grand LAgrangian Deployment
(GLAD) dataset as a ragged-array Xarray Dataset.

The dataset and its description are hosted at https://doi.org/10.7266/N7VD6WC8.

Example
-------
>>> from clouddrift.adapters import glad
>>> ds = glad.to_xarray()

Reference
---------
Özgökmen, Tamay. 2013. GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records), northern Gulf of Mexico near DeSoto Canyon, July-October 2012. Distributed by: Gulf of Mexico Research Initiative Information and Data Cooperative (GRIIDC), Harte Research Institute, Texas A&M University–Corpus Christi. doi:10.7266/N7VD6WC8
"""

from io import BytesIO

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.adapters.utils import download_with_progress


def get_dataframe() -> pd.DataFrame:
    """Get the GLAD dataset as a pandas DataFrame."""
    url = "https://data.gulfresearchinitiative.org/pelagos-symfony/api/file/download/169841"
    # GRIIDC server doesn't provide Content-Length header, so we'll hardcode
    # the expected data length here.
    file_size = 155330876
    buf = BytesIO(b"")
    download_with_progress([(url, buf, file_size)])
    buf.seek(0)
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
    """Return the GLAD data as a ragged-array Xarray Dataset."""
    df = get_dataframe()
    ds = df.to_xarray()

    traj, rowsize = np.unique(ds.id, return_counts=True)

    # Make the dataset compatible with clouddrift functions.
    ds = (
        ds.swap_dims({"index": "obs"})
        .drop_vars(["id", "index"])
        .assign_coords(traj=traj)
        .assign({"rowsize": ("traj", rowsize)})
        .rename_vars({"obs": "time", "traj": "id"})
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

    # Set global attributes
    ds.attrs = {
        "title": "GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records), northern Gulf of Mexico near DeSoto Canyon, July-October 2012",
        "institution": "Consortium for Advanced Research on Transport of Hydrocarbon in the Environment (CARTHE)",
        "source": "CODE-style drifters",
        "history": "Downloaded from https://data.gulfresearchinitiative.org/data/R1.x134.073:0004 and post-processed into a ragged-array Xarray Dataset by CloudDrift",
        "references": "Özgökmen, Tamay. 2013. GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records), northern Gulf of Mexico near DeSoto Canyon, July-October 2012. Distributed by: Gulf of Mexico Research Initiative Information and Data Cooperative (GRIIDC), Harte Research Institute, Texas A&M University–Corpus Christi. doi:10.7266/N7VD6WC8",
    }

    return ds

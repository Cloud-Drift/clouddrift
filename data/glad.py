import numpy as np
import pandas as pd
import xarray as xr
import urllib.request
import os
from os.path import isfile, join, exists
from datetime import datetime

# GRIIDC https://data.gulfresearchinitiative.org/
# GLAD experiment CODE-style drifter trajectories (low-pass filtered, 15 minute interval records)
# https://data.gulfresearchinitiative.org/data/R1.x134.073:0004
griidc_url = (
    "https://data.gulfresearchinitiative.org/pelagos-symfony/api/file/download/169841"
)
folder = "../data/raw/glad/"
file = "GLAD_15min_filtered.dat"
os.makedirs(folder, exist_ok=exists(folder))  # create raw data folder

# download and parse the file
if not isfile(join(folder, file)):
    req = urllib.request.urlretrieve(griidc_url, join(folder, file))
    print(f"Dataset {file} downloaded to '{folder}'.")
else:
    print(f"Dataset {file} already in '{folder}'.")

# parse the csv file
df = pd.read_csv(
    join(folder, file),
    delimiter="\s+",
    header=5,
    names=["id", "date", "time", "lat", "lon", "err_pos", "ve", "vn", "err_vel"],
)
df.insert(0, "datetime", pd.to_datetime(df["date"] + " " + df["time"]))
df["datetime"] = [(t - datetime(1970, 1, 1)).total_seconds() for t in df["datetime"]]
df = df.drop(labels=["date", "time"], axis=1)
df.id = pd.to_numeric(df.id.str.slice(start=-3))
df = df.set_index("id")


def preprocess(index: int) -> xr.Dataset:
    """
    Extract the Lagrangian data for one trajectory from a pd.Dataframe

    :param index: drifter's identification number
    :return: xr.Dataset containing the data and attributes
    """
    df_subset = df.loc[index]
    rowsize = len(df_subset)

    return xr.Dataset(
        data_vars=dict(
            ID=(["traj"], [index], {"long_name": "Buoy ID", "units": "-"}),
            rowsize=(
                ["traj"],
                [rowsize],
                {"long_name": "Number of observations per trajectory", "units": "-"},
            ),
            err_pos=(
                ["obs"],
                df_subset.err_pos,
                {"long_name": "estimated position error", "units": "m"},
            ),
            ve=(
                ["obs"],
                df_subset.ve,
                {"long_name": "Eastward velocity", "units": "m/s"},
            ),
            vn=(
                ["obs"],
                df_subset.vn,
                {"long_name": "Northward velocity", "units": "m/s"},
            ),
            err_vel=(
                ["obs"],
                df_subset.err_vel,
                {"long_name": "Standard error in latitude", "units": "degrees_north"},
            ),
        ),
        coords=dict(
            longitude=(
                ["obs"],
                df_subset.lon,
                {"long_name": "Longitude", "units": "degrees_east"},
            ),
            latitude=(
                ["obs"],
                df_subset.lat,
                {"long_name": "Latitude", "units": "degrees_north"},
            ),
            time=(
                ["obs"],
                df_subset.datetime,
                {"long_name": "Time", "units": "seconds since 1970-01-01 00:00:00"},
            ),
            ids=(
                ["obs"],
                df_subset.index,
                {"long_name": "Buoy ID for all observations", "units": "-"},
            ),
        ),
        attrs={
            "title": "Glad experiment",
        },
    )

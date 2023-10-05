"""
This module defines functions used to adapt the MOSAiC sea-ice drift dataset as
a ragged-array dataset.

The dataset is hosted at https://doi.org/10.18739/A2KP7TS83.

Reference: Angela Bliss, Jennifer Hutchings, Philip Anderson, Philipp Anhaus,
Hans Jakob Belter, Jørgen Berge, Vladimir Bessonov, Bin Cheng, Sylvia Cole,
Dave Costa, Finlo Cottier, Christopher J Cox, Pedro R De La Torre, Dmitry V Divine,
Gilbert Emzivat, Ying-Chih Fang, Steven Fons, Michael Gallagher, Maxime Geoffrey,
Mats A Granskog, ... Guangyu Zuo. (2022). Sea ice drift tracks from the Distributed
Network of autonomous buoys deployed during the Multidisciplinary drifting Observatory
for the Study of Arctic Climate (MOSAiC) expedition 2019 - 2021. Arctic Data Center.
doi:10.18739/A2KP7TS83.

Example
-------
>>> from clouddrift.adapters import mosaic
>>> ds = mosaic.to_xarray()
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import xarray as xr
import xml.etree.ElementTree as ET


def get_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get the MOSAiC data (obs dimension in the target Dataset) and metadata
    (traj dimension in the target dataset ) as pandas DataFrames."""
    xml = get_repository_metadata()
    filenames, urls = get_file_urls(xml)
    exclude_patterns = ["site_buoy_summary", "buoy_list"]
    data_filenames = [
        f for f in filenames if not any([s in f for s in exclude_patterns])
    ]
    data_urls = [
        f
        for n, f in enumerate(urls)
        if not any([s in filenames[n] for s in exclude_patterns])
    ]
    sensor_ids = [f.split("_")[-1].rstrip(".csv") for f in data_filenames]
    sensor_list_url = urls[
        filenames.index([f for f in filenames if "buoy_list" in f].pop())
    ]
    sensors = pd.read_csv(sensor_list_url)

    # Sort the urls by the order of sensor IDs in the sensor list
    order_index = {id: n for n, id in enumerate(sensors["Sensor ID"])}
    sorted_indices = sorted(
        range(len(sensor_ids)), key=lambda k: order_index[sensor_ids[k]]
    )
    sorted_data_urls = [data_urls[i] for i in sorted_indices]

    with ThreadPoolExecutor() as executor:
        dfs = tqdm(
            executor.map(pd.read_csv, sorted_data_urls),
            total=len(sorted_data_urls),
            desc="Downloading data",
            ncols=80,
        )

    obs_df = pd.concat(dfs)

    # Use the index of the concatenated DataFrame to determine the count/rowsize
    zero_indices = [n for n, val in enumerate(list(obs_df.index)) if val == 0]
    sensors["rowsize"] = np.diff(zero_indices + [len(obs_df)])

    # Make the time column the index of the DataFrame, which will make it a
    # coordinate in the xarray Dataset.
    obs_df.set_index("datetime", inplace=True)
    sensors.set_index("Sensor ID", inplace=True)

    return obs_df, sensors


def get_file_urls(xml: str) -> list[str]:
    """Pass the MOSAiC XML string and return the list of filenames and URLs."""
    filenames = [
        tag.text
        for tag in ET.fromstring(xml).findall("./dataset/dataTable/physical/objectName")
    ]
    urls = [
        tag.text
        for tag in ET.fromstring(xml).findall(
            "./dataset/dataTable/physical/distribution/online/url"
        )
    ]
    return filenames, urls


def get_repository_metadata() -> str:
    """Get the MOSAiC repository metadata as an XML string.
    Pass this string to other get_* functions to extract the data you need.
    """
    url = "https://arcticdata.io/metacat/d1/mn/v2/object/doi:10.18739/A2KP7TS83"
    r = requests.get(url)
    return r.content


def to_xarray():
    """Return the MOSAiC data as an ragged-array Xarray Dataset."""

    # Download the data and metadata as pandas DataFrames.
    obs_df, traj_df = get_dataframes()

    # Dates and datetimes are strings; convert them to datetime64 instances
    # for compatibility with CloudDrift's analysis functions.
    obs_df.index = pd.to_datetime(obs_df.index)
    for col in [
        "Deployment Date",
        "Deployment Datetime",
        "First Data Datetime",
        "Last Data Datetime",
    ]:
        traj_df[col] = pd.to_datetime(traj_df[col])

    # Merge into an Xarray Dataset and rename the dimensions and variables to
    # follow the CloudDrift convention.
    ds = xr.merge([obs_df.to_xarray(), traj_df.to_xarray()])
    ds = ds.rename_dims({"datetime": "obs", "Sensor ID": "traj"}).rename_vars(
        {"datetime": "time", "Sensor ID": "id"}
    )

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

    return ds

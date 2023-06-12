"""
This module provides functions to easily access ragged-array datasets.
"""

import xarray as xr


def gdp1h() -> xr.Dataset:
    """
    Returns the hourly GDP dataset as an Xarray dataset.

    Returns
    -------
    xarray.Dataset
        GDP1h dataset
    """
    url = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp_v2.00.zarr"
    return xr.open_dataset(url, engine="zarr")

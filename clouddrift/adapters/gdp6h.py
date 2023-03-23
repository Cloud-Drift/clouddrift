"""
This module provides functions and metadata that can be used to convert the
6-hourly Global Drifter Program (GDP) data to a ``clouddrift.RaggedArray``
instance.
"""

from ..dataformat import RaggedArray
import numpy as np
import urllib.request
import concurrent.futures
import re
import tempfile
from tqdm import tqdm
from typing import Optional
import os
import warnings

import clouddrift.adapters.gdp as gdp


GDP_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/netcdf/"
GDP_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdp6h")


def download(
    drifter_ids: list = None, n_random_id: int = None, url: str = GDP_DATA_URL
):
    """Download individual NetCDF files from the AOML server.

    Parameters
    ----------
    drifter_ids : list
        List of drifter to retrieve (Default: all)
    n_random_id : int
        Randomly select n_random_id drifter IDs to download (Default: None)
    url : str
        URL from which to download the data (Default: GDP_DATA_URL). Alternatively, it can be GDP_DATA_URL_EXPERIMENTAL.

    Returns
    -------
    out : list
        List of retrived drifters
    """

    # Create a temporary directory if doesn't already exists.
    os.makedirs(GDP_TMP_PATH, exist_ok=True)

    pattern = "drifter_[0-9]*.nc"
    directory_list = [
        "buoydata_1_5000",
        "buoydata_5001_10000",
        "buoydata_10001_15000",
        "buoydata_15001_oct22",
    ]

    # retrieve all drifter ID numbers
    if drifter_ids is None:
        urlpath = urllib.request.urlopen(url)
        string = urlpath.read().decode("utf-8")
        drifter_urls = []
        for dir in directory_list:
            urlpath = urllib.request.urlopen(os.path.join(url, dir))
            string = urlpath.read().decode("utf-8")
            filelist = list(set(re.compile(pattern).findall(string)))
            drifter_urls += [os.path.join(url, dir, f) for f in filelist]

    # retrieve only a subset of n_random_id trajectories
    if n_random_id:
        if n_random_id > len(drifter_urls):
            warnings.warn(
                f"Retrieving all listed trajectories because {n_random_id} is larger than the {len(drifter_ids)} listed trajectories."
            )
        else:
            rng = np.random.RandomState(42)
            drifter_urls = rng.choice(drifter_urls, n_random_id, replace=False)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Asynchronously download individual netCDF files
        list(
            tqdm(
                executor.map(
                    gdp.fetch_netcdf,
                    drifter_urls,
                    [
                        os.path.join(GDP_TMP_PATH, os.path.basename(f))
                        for f in drifter_urls
                    ],
                ),
                total=len(drifter_urls),
                desc="Downloading files",
                ncols=80,
            )
        )

    # Download the metadata so we can order the drifter IDs by end date.
    gdp_metadata = gdp.get_gdp_metadata()
    drifter_ids = [
        int(os.path.basename(f).split("_")[1].split(".")[0]) for f in drifter_urls
    ]

    return gdp.order_by_date(gdp_metadata, drifter_ids)


def to_raggedarray(
    drifter_ids: Optional[list[int]] = None,
    n_random_id: Optional[int] = None,
    url: Optional[str] = GDP_DATA_URL,
) -> RaggedArray:
    """Download and process individual GDP hourly files and return a RaggedArray
    instance with the data.

    Parameters
    ----------
    drifter_ids : list[int], optional
        List of drifters to retrieve (Default: all)
    n_random_id : list[int], optional
        Randomly select n_random_id drifter NetCDF files

    Returns
    -------
    out : RaggedArray
        A RaggedArray instance of the requested dataset
    """
    ids = download(drifter_ids, n_random_id, url)

    return RaggedArray.from_files(
        indices=ids,
        preprocess_func=gdp.preprocess,
        name_coords=gdp.GDP_COORDS,
        name_meta=gdp.GDP_METADATA,
        name_data=gdp.GDP_DATA,
        rowsize_func=gdp.rowsize,
        filename_pattern="drifter_{id}.nc",
    )

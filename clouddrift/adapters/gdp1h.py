"""
This module provides functions and metadata that can be used to convert the
hourly Global Drifter Program (GDP) data to a ``clouddrift.RaggedArray``
instance.
"""

import clouddrift.adapters.gdp as gdp
from clouddrift.dataformat import RaggedArray
import numpy as np
import urllib.request
import concurrent.futures
import re
import tempfile
from tqdm import tqdm
from typing import Optional
import os
import warnings


GDP_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/v2.00/netcdf/"
GDP_DATA_URL_EXPERIMENTAL = (
    "https://www.aoml.noaa.gov/ftp/pub/phod/lumpkin/hourly/experimental/"
)
GDP_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdp")


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

    print(f"Downloading GDP hourly data to {GDP_TMP_PATH}...")

    # Create a temporary directory if doesn't already exists.
    os.makedirs(GDP_TMP_PATH, exist_ok=True)

    if url == GDP_DATA_URL:
        pattern = "drifter_[0-9]*.nc"
        filename_pattern = "drifter_{id}.nc"
    elif url == GDP_DATA_URL_EXPERIMENTAL:
        pattern = "drifter_hourly_[0-9]*.nc"
        filename_pattern = "drifter_hourly_{id}.nc"

    # retrieve all drifter ID numbers
    if drifter_ids is None:
        urlpath = urllib.request.urlopen(url)
        string = urlpath.read().decode("utf-8")
        filelist = re.compile(pattern).findall(string)
        drifter_ids = np.unique([int(f.split("_")[-1][:-3]) for f in filelist])

    # retrieve only a subset of n_random_id trajectories
    if n_random_id:
        if n_random_id > len(drifter_ids):
            warnings.warn(
                f"Retrieving all listed trajectories because {n_random_id} is larger than the {len(drifter_ids)} listed trajectories."
            )
        else:
            rng = np.random.RandomState(42)
            drifter_ids = sorted(rng.choice(drifter_ids, n_random_id, replace=False))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # create list of urls and paths
        urls = []
        files = []
        for i in drifter_ids:
            file = filename_pattern.format(id=i)
            urls.append(os.path.join(url, file))
            files.append(os.path.join(GDP_TMP_PATH, file))

        # parallel retrieving of individual netCDF files
        list(
            tqdm(
                executor.map(gdp.fetch_netcdf, urls, files),
                total=len(files),
                desc="Downloading files",
                ncols=80,
            )
        )

    # Download the metadata so we can order the drifter IDs by end date.
    gdp_metadata = gdp.get_gdp_metadata()

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
    url : str, optional
        URL from which to download the data (Default: GDP_DATA_URL).
        Alternatively, it can be GDP_DATA_URL_EXPERIMENTAL.

    Returns
    -------
    out : RaggedArray
        A RaggedArray instance of the requested dataset
    """
    ids = download(drifter_ids, n_random_id, url)

    if url == GDP_DATA_URL:
        filename_pattern = "drifter_{id}.nc"
    elif url == GDP_DATA_URL_EXPERIMENTAL:
        filename_pattern = "drifter_hourly_{id}.nc"
    else:
        raise ValueError(f"url must be {GDP_DATA_URL} or {GDP_DATA_URL_EXPERIMENTAL}.")

    return RaggedArray.from_files(
        indices=ids,
        preprocess_func=gdp.preprocess,
        name_coords=gdp.GDP_COORDS,
        name_meta=gdp.GDP_METADATA,
        name_data=gdp.GDP_DATA,
        rowsize_func=gdp.rowsize,
        filename_pattern=filename_pattern,
        tmp_path=GDP_TMP_PATH,
    )

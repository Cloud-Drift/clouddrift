"""
This module defines functions used to adapt the MOSAiC sea-ice drift dataset as
a ragged-array dataset.
"""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET


def get_metadata() -> str:
    """Get the MOSAiC metadata as an XML string.
    Pass this string to other get_* functions to extract the data you need.
    """
    url = "https://arcticdata.io/metacat/d1/mn/v2/object/doi:10.18739/A2KP7TS83"
    r = requests.get(url)
    return r.content


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
    return [{"filename": f, "url": u} for f, u in zip(filenames, urls)]


def get_dataframe() -> pd.DataFrame:
    """Get the MOSAiC data as a pandas DataFrame."""
    xml = get_metadata()
    urls = [
        f["url"]
        for f in get_file_urls(xml)
        if "site_buoy_summary" not in f["filename"] and "buoy_list" not in f["filename"]
    ]
    with ThreadPoolExecutor() as executor:
        dfs = tqdm(
            executor.map(pd.read_csv, urls),
            total=len(urls),
            desc="Downloading data",
            ncols=80,
        )
    # TODO collect ids from filenames
    # TODO insert ids as a column
    return pd.concat(dfs)

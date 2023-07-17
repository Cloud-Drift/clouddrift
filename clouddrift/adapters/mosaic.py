"""
This module defines functions used to adapt the MOSAiC sea-ice drift dataset as
a ragged-array dataset.
"""
import pandas as pd
import requests
import xml.etree.ElementTree as ET


def get_metadata() -> str:
    """Get the MOSAiC metadata as an XML string.
    Pass this string to other get_* functions to extract the data you need.
    """
    url = "https://arcticdata.io/metacat/d1/mn/v2/object/doi:10.18739/A2KP7TS83"
    r = requests.get(url)
    return r.content


def get_filenames(xml: str) -> list[str]:
    """Pass the MOSAiC XML string and return the list of available filenames."""
    return [
        tag.text
        for tag in ET.fromstring(xml).findall("./dataset/dataTable/physical/objectName")
    ]


def get_file_urls(xml: str) -> list[str]:
    """Pass the MOSAiC XML string and return the list of URLs for the available files."""
    return [
        tag.text
        for tag in ET.fromstring(xml).findall(
            "./dataset/dataTable/physical/distribution/online/url"
        )
    ]

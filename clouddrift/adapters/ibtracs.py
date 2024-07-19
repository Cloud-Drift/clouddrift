from typing import Literal, TypeAlias

from fsspec.implementations.http import HTTPFileSystem

SOURCE_BASE_URI = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"

SOURCE_URL_FMT = "{base_uri}/{version}/access/netcdf/IBTrACS.{kind}.{version}.nc"

_Version: TypeAlias = Literal["v03r09"] | Literal["v04r00"] | Literal["v04r01"]

_Kind: TypeAlias = (
    Literal["ACTIVE"]
    | Literal["ALL"]
    | Literal["EP"]
    | Literal["NA"]
    | Literal["NI"]
    | Literal["SA"]
    | Literal["SI"]
    | Literal["SP"]
    | Literal["WP"]
    | Literal["LAST_3_YEARS"]
    | Literal["SINCE_1980"]
)


def download(version: _Version, kind: _Kind):
    fs = HTTPFileSystem()
    return fs.open(_get_source_url(version, kind), callback=lambda: print("what"))


def _kind_map(kind: _Kind):
    return {"LAST_3_YEARS": "last3years", "SINCE_1980": "since1980"}.get(kind, kind)


def _get_source_url(version: _Version, kind: _Kind):
    return SOURCE_URL_FMT.format(
        base_uri=SOURCE_BASE_URI, version=version, kind=_kind_map(kind)
    )

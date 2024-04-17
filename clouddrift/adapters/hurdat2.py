"""
This module defines functions used to adapt the HURDAT2 cyclone track data as
a ragged-array dataset.
"""

import enum
import os
import re
import tempfile
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from io import StringIO
from typing import Literal

import numpy as np
import xarray as xr

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

_DEFAULT_NAME = "hurdat2"
_ATLANTIC_BASIN_URL = "https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html"
_PACIFIC_BASIN_URL = "https://www.aoml.noaa.gov/hrd/hurdat/hurdat2-nepac.html"

_DEFAULT_FILE_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", _DEFAULT_NAME)

_METERS_IN_NAUTICAL_MILES = 1825
_PASCAL_PER_MILLIBAR = 100

_BasinOption = Literal["atlantic", "pacific", "both"]


class RecordIdentifier(str, enum.Enum):
    """
    C – Closest approach to a coast, not followed by a landfall
    G – Genesis
    I – An intensity peak in terms of both pressure and wind
    L – Landfall (center of system crossing a coastline)
    P – Minimum in central pressure
    R – Provides additional detail on the intensity of the cyclone when rapid changes are underway
    S – Change of status of the system
    T – Provides additional detail on the track (position) of the cyclone
    W – Maximum sustained wind speed
    """

    CLOSEST_TO_COAST = "C"
    GENESIS = "G"
    INTENSITY_PEAK = "I"
    LANDFALL = "L"
    MINIMUM_CENTRAL_PRESSURE = "P"
    RAPID_CHANGE = "R"
    STATUS_CHANGE = "S"
    TRACK_POSITION = "T"
    MAX_SUSTAINED_WIND_SPEED = "W"
    NOT_AVAILABLE = ""


class SystemStatus(str, enum.Enum):
    """
    TD – Tropical cyclone of tropical depression intensity (< 34 knots)
    TS – Tropical cyclone of tropical storm intensity (34-63 knots)
    HU – Tropical cyclone of hurricane intensity (> 64 knots)
    EX – Extratropical cyclone (of any intensity)
    SD – Subtropical cyclone of subtropical depression intensity (< 34 knots)
    SS – Subtropical cyclone of subtropical storm intensity (> 34 knots)
    LO – A low that is neither a tropical cyclone, a subtropical cyclone, nor an extratropical cyclone (of any intensity)
    WV – Tropical Wave (of any intensity)
    DB – Disturbance (of any intensity)
    ET - UNKNOWN found in Northeast Pacific Basin
    PT - UNKNOWN found in Northeast Pacific Basin
    ST - UNKNOWN found in Northeast Pacific Basin
    TY - UNKNOWN found in Northeast Pacific Basin
    """

    TD = "TD"
    TS = "TS"
    HU = "HU"
    EX = "EX"
    SD = "SD"
    SS = "SS"
    LO = "LO"
    WV = "WV"
    DB = "DB"
    ET = "ET"
    PT = "PT"
    ST = "ST"
    TY = "TY"


@dataclass
class HeaderLine:
    id: str = field(
        metadata={
            "standard_name": "automated_tropical_cyclone_forecasting_system_storm_identifier",
            "comment": "The Automated Tropical Cyclone Forecasting System (ATCF) storm identifier is an 8 character string which identifies a tropical cyclone. The storm identifier has the form BBCCYYYY, where BB is the ocean basin, specifically: AL - North Atlantic basin, north of the Equator; SL - South Atlantic basin, south of the Equator; EP - North East Pacific basin, eastward of 140 degrees west longitude; CP - North Central Pacific basin, between the dateline and 140 degrees west longitude; WP - North West Pacific basin, westward of the dateline; IO - North Indian Ocean basin, north of the Equator between 40 and 100 degrees east longitude; SH - South Pacific Ocean basin and South Indian Ocean basin. CC is the cyclone number. Numbers 01 through 49 are reserved for tropical and subtropical cyclones. A cyclone number is assigned to each tropical or subtropical cyclone in each basin as it develops. Numbers are assigned in chronological order. Numbers 50 through 79 are reserved for internal use by operational forecast centers. Numbers 80 through 89 are reserved for training, exercises and testing. Numbers 90 through 99 are reserved for tropical disturbances having the potential to become tropical or subtropical cyclones. The 90's are assigned sequentially and reused throughout the calendar year. YYYY is the four-digit year. This is calendar year for the northern hemisphere. For the southern hemisphere, the year begins July 1, with calendar year plus one. Reference: Miller, R.J., Schrader, A.J., Sampson, C.R., & Tsui, T.L. (1990), The Automated Tropical Cyclone Forecasting System (ATCF), American Meteorological Society Computer Techniques, 5, 653 - 660.",
        }
    )
    basin: str = field(
        metadata={
            "comments": "Basin of origin, possible values: AL - North Atlantic basin, north of the Equator; SL - South Atlantic basin, south of the Equator; EP - North East Pacific basin, eastward of 140 degrees west longitude; CP - North Central Pacific basin, between the dateline and 140 degrees west longitude; WP - North West Pacific basin, westward of the dateline; IO - North Indian Ocean basin, north of the Equator between 40 and 100 degrees east longitude; SH - South Pacific Ocean basin and South Indian Ocean basin",
        }
    )
    year: int = field(metadata={"long_name": "Year"})
    rowsize: int = field(metadata={"comment": "Number of best track entries"})


@dataclass
class DataLine:
    time: datetime = field(
        metadata={"comments": "Computed property from YYY-MM-DD HH:MM in UTC"}
    )
    record_identifier: RecordIdentifier = field(
        metadata={
            "standard_name": "Record Idenfier",
            "comments": RecordIdentifier.__doc__,
        }
    )
    system_status: SystemStatus = field(
        metadata={"standard_name": "System Status", "comments": SystemStatus.__doc__}
    )
    lat: float = field(
        metadata={
            "standard_name": "latitude",
            "units": "degree_north",
        }
    )
    lon: float = field(
        metadata={
            "standard_name": "longitude",
            "units": "degree_east",
        }
    )
    wind_speed: float = field(
        metadata={
            "standard_name": "wind_speed",
            "units": "m s-1",
        }
    )
    pressure: float = field(
        metadata={
            "standard_name": "air_pressure",
            "units": "Pa",
        }
    )
    max_low_wind_radius_ne: float = field(
        metadata={
            "comment": "34 kt Wind maximum radius in the storms NE Quadrant",
            "units": "m",
        }
    )
    max_low_wind_radius_se: float = field(
        metadata={
            "comment": "34 kt Wind maximum radius in the storms SE Quadrant",
            "units": "m",
        }
    )
    max_low_wind_radius_sw: float = field(
        metadata={
            "comment": "34 kt Wind maximum radius in the storms SW Quadrant",
            "units": "m",
        }
    )
    max_low_wind_radius_nw: float = field(
        metadata={
            "comment": "34 kt Wind maximum radius in the storms NW Quadrant",
            "units": "m",
        }
    )
    max_med_wind_radius_ne: float = field(
        metadata={
            "comment": "50 kt Wind maximum radius in the storms NE Quadrant",
            "units": "m",
        }
    )
    max_med_wind_radius_se: float = field(
        metadata={
            "comment": "50 kt Wind maximum radius in the storms SE Quadrant",
            "units": "m",
        }
    )
    max_med_wind_radius_sw: float = field(
        metadata={
            "comment": "50 kt Wind maximum radius in the storms SW Quadrant",
            "units": "m",
        }
    )
    max_med_wind_radius_nw: float = field(
        metadata={
            "comment": "50 kt Wind maximum radius in the storms NW Quadrant",
            "units": "m",
        }
    )
    max_high_wind_radius_ne: float = field(
        metadata={
            "comment": "64 kt Wind maximum radius in the storms NE Quadrant",
            "units": "m",
        }
    )
    max_high_wind_radius_se: float = field(
        metadata={
            "comment": "64 kt Wind maximum radius in the storms SE Quadrant",
            "units": "m",
        }
    )
    max_high_wind_radius_sw: float = field(
        metadata={
            "comment": "64 kt Wind maximum radius in the storms SW Quadrant",
            "units": "m",
        }
    )
    max_high_wind_radius_nw: float = field(
        metadata={
            "comment": "64 kt Wind maximum radius in the storms NW Quadrant",
            "units": "m",
        }
    )
    max_sustained_wind_speed_radius: float = field(
        metadata={
            "standard_name": "radius_of_tropical_cyclone_maximum_sustained_wind_speed",
            "units": "m",
        }
    )


@dataclass
class TrackData:
    global_attrs = {
        "title": "HURricane DATa 2nd generation (HURDAT2)",
        "date_created": datetime.now().isoformat(),
        "publisher_name": "NOAA AOML Hurricane Research Division",
        "publisher_email": "AOML.HRDWebmaster@noaa.gov",
        "publisher_url": "https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html",
        "institution": "NOAA Atlantic Oceanographic and Meteorological Laboratory",
        "summary": "The National Hurricane Center (NHC) conducts a post-storm analysis of each tropical cyclone in its area of responsibility to determine the official assessment of the cyclone's history",
    }

    header: HeaderLine
    data: list[DataLine]

    def get_rowsize(self) -> int:
        return len(self.data)

    def to_xarray_dataset(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "basin": (["traj"], np.array([self.header.basin])),
                "year": (["traj"], np.array([self.header.year])),
                "rowsize": (["traj"], np.array([self.header.rowsize])),
                "record_identifier": (
                    ["obs"],
                    np.array([line.record_identifier for line in self.data]),
                ),
                "system_status": (
                    ["obs"],
                    np.array([line.system_status for line in self.data]),
                ),
                "lat": (["obs"], np.array([line.lat for line in self.data])),
                "lon": (["obs"], np.array([line.lon for line in self.data])),
                "wind_speed": (
                    ["obs"],
                    np.array([line.wind_speed for line in self.data]),
                ),
                "pressure": (["obs"], np.array([line.pressure for line in self.data])),
                "max_low_wind_radius_ne": (
                    ["obs"],
                    np.array([line.max_low_wind_radius_ne for line in self.data]),
                ),
                "max_low_wind_radius_se": (
                    ["obs"],
                    np.array([line.max_low_wind_radius_se for line in self.data]),
                ),
                "max_low_wind_radius_sw": (
                    ["obs"],
                    np.array([line.max_low_wind_radius_sw for line in self.data]),
                ),
                "max_low_wind_radius_nw": (
                    ["obs"],
                    np.array([line.max_low_wind_radius_nw for line in self.data]),
                ),
                "max_med_wind_radius_ne": (
                    ["obs"],
                    np.array([line.max_med_wind_radius_ne for line in self.data]),
                ),
                "max_med_wind_radius_se": (
                    ["obs"],
                    np.array([line.max_med_wind_radius_se for line in self.data]),
                ),
                "max_med_wind_radius_sw": (
                    ["obs"],
                    np.array([line.max_med_wind_radius_sw for line in self.data]),
                ),
                "max_med_wind_radius_nw": (
                    ["obs"],
                    np.array([line.max_med_wind_radius_nw for line in self.data]),
                ),
                "max_high_wind_radius_ne": (
                    ["obs"],
                    np.array([line.max_high_wind_radius_ne for line in self.data]),
                ),
                "max_high_wind_radius_se": (
                    ["obs"],
                    np.array([line.max_high_wind_radius_se for line in self.data]),
                ),
                "max_high_wind_radius_sw": (
                    ["obs"],
                    np.array([line.max_high_wind_radius_sw for line in self.data]),
                ),
                "max_high_wind_radius_nw": (
                    ["obs"],
                    np.array([line.max_high_wind_radius_nw for line in self.data]),
                ),
                "max_sustained_wind_speed_radius": (
                    ["obs"],
                    np.array(
                        [line.max_sustained_wind_speed_radius for line in self.data]
                    ),
                ),
            },
            coords={
                "id": (["traj"], np.array([self.header.id])),
                "time": (
                    ["obs"],
                    np.array(
                        [int(line.time.timestamp() * 10**9) for line in self.data],
                        dtype="datetime64[ns]",
                    ),
                ),
            },
        )


def _map_heading(coordinate: str) -> float:
    heading_map = {"N": 1, "E": 1, "S": -1, "W": -1}
    cardinal_direction = coordinate[-1]
    heading = coordinate[:-1]

    if (sign := heading_map.get(cardinal_direction)) is not None:
        return float(heading) * sign
    raise ValueError(f"Invalid cardinal direction: {cardinal_direction}")


def _get_download_requests(basin: _BasinOption, tmp_path: str):
    download_requests: list[tuple[str, str, None]] = list()

    if basin == "atlantic" or basin == "both":
        file_name = _ATLANTIC_BASIN_URL.split("/")[-1]
        fp = os.path.join(_DEFAULT_FILE_PATH, file_name)
        download_requests.append((_ATLANTIC_BASIN_URL, fp, None))

    if basin == "pacific" or basin == "both":
        file_name = _PACIFIC_BASIN_URL.split("/")[-1]
        fp = os.path.join(_DEFAULT_FILE_PATH, file_name)
        download_requests.append((_PACIFIC_BASIN_URL, fp, None))
    return download_requests


def to_raggedarray(
    basin: _BasinOption = "both",
    tmp_path: str = _DEFAULT_FILE_PATH,
    convert: bool = True,
) -> RaggedArray:
    os.makedirs(
        _DEFAULT_FILE_PATH, exist_ok=True
    )  # generate temp directory for hurdat related intermerdiary data
    download_requests = _get_download_requests(basin, tmp_path)
    download_with_progress(download_requests)
    track_data = list()

    for _, fp, _ in download_requests:
        track_data.extend(_extract_track_data(fp, convert))

    metadata_fields = list()
    data_fields = list()

    for f in fields(HeaderLine):
        if f.name != "id":
            metadata_fields.append(f.name)

    for f in fields(DataLine):
        if f.name != "time":
            data_fields.append(f.name)

    ra = RaggedArray.from_files(
        indices=list(range(0, len(track_data))),
        name_coords=["id", "time"],
        name_meta=metadata_fields,
        name_data=data_fields,
        name_dims={"traj": "rows", "obs": "obs"},
        rowsize_func=lambda idx: track_data[idx].get_rowsize(),
        preprocess_func=lambda idx: track_data[idx].to_xarray_dataset(),
        attrs_global=TrackData.global_attrs,
        attrs_variables={
            field.name: field.metadata
            for field in fields(HeaderLine) + fields(DataLine)
        },
    )
    return ra


def _apply_or_nan(val: str, cond: bool, func) -> float:
    """If `val` is None or condition is true return nan otherwise apply the `func` to `val` and return the result"""
    if val is None or cond:
        return np.nan
    return func(val)


def _extract_track_data(datafile_path: str, convert: bool) -> list[TrackData]:
    datapage = StringIO()
    with open(datafile_path, "rb") as file:
        while (data := file.read()) != b"":
            datapage.write(data.decode("utf-8").replace(" ", ""))
    datapage.seek(0)

    data_line_count = 0
    current_header = None
    data_lines = list()
    track_data = list[TrackData]()

    is_html_line = lambda line: re.match(r"[html|head|pre]+", line)
    is_header_line = (
        lambda cols, data_line_count: len(cols) == 4 and data_line_count == 0
    )
    is_data_line = lambda cols, data_line_count: len(cols) == 21 and data_line_count > 0
    if convert:
        nm_to_m = (
            lambda x: float(x) * _METERS_IN_NAUTICAL_MILES
        )  # nautical-miles to meters
        k_to_mps = (
            lambda x: float(x) * _METERS_IN_NAUTICAL_MILES / 3600
        )  # knots to meters per second
        mb_to_pa = lambda x: float(x) * _PASCAL_PER_MILLIBAR  # millibar to pascal
    else:
        nm_to_m = lambda x: float(x)
        k_to_mps = lambda x: float(x)
        mb_to_pa = lambda x: float(x)

    while (line := datapage.readline()) != "":
        if is_html_line(line) or line == "\r\n":
            if current_header is not None:
                track_data.append(TrackData(current_header, data_lines))
                data_lines = list()
                current_header = None
            continue

        cols = line.split(",")

        if is_header_line(cols, data_line_count):
            data_line_count = int(cols[2])
            header = HeaderLine(
                id=cols[0],
                basin=cols[0][:2],
                year=int(cols[0][4:8]),
                rowsize=data_line_count,
            )
            if current_header is None:
                current_header = header
            else:
                track_data.append(TrackData(current_header, data_lines))
                data_lines = list()
                current_header = header
        elif (
            is_data_line(cols, data_line_count)
            and len(cols[0]) == 8
            and len(cols[1]) == 4
        ):
            data_lines.append(
                DataLine(
                    time=datetime(
                        year=int(cols[0][:4]),
                        month=int(cols[0][4:6]),
                        day=int(cols[0][6:8]),
                        hour=int(cols[1][:2]),
                        minute=int(cols[1][2:4]),
                        tzinfo=timezone.utc,
                    ),
                    record_identifier=RecordIdentifier(cols[2]),
                    system_status=SystemStatus(cols[3]),
                    lat=_map_heading(cols[4]),
                    lon=_map_heading(cols[5]),
                    wind_speed=_apply_or_nan(cols[6], float(cols[6]) < 0, k_to_mps),
                    pressure=_apply_or_nan(cols[7], cols[7] == "-999", mb_to_pa),
                    max_low_wind_radius_ne=_apply_or_nan(
                        cols[8], cols[8] == "-999", nm_to_m
                    ),
                    max_low_wind_radius_se=_apply_or_nan(
                        cols[9], cols[9] == "-999", nm_to_m
                    ),
                    max_low_wind_radius_sw=_apply_or_nan(
                        cols[10], cols[10] == "-999", nm_to_m
                    ),
                    max_low_wind_radius_nw=_apply_or_nan(
                        cols[11], cols[11] == "-999", nm_to_m
                    ),
                    max_med_wind_radius_ne=_apply_or_nan(
                        cols[12], cols[12] == "-999", nm_to_m
                    ),
                    max_med_wind_radius_se=_apply_or_nan(
                        cols[13], cols[13] == "-999", nm_to_m
                    ),
                    max_med_wind_radius_sw=_apply_or_nan(
                        cols[14], cols[14] == "-999", nm_to_m
                    ),
                    max_med_wind_radius_nw=_apply_or_nan(
                        cols[15], cols[15] == "-999", nm_to_m
                    ),
                    max_high_wind_radius_ne=_apply_or_nan(
                        cols[16], cols[16] == "-999", nm_to_m
                    ),
                    max_high_wind_radius_se=_apply_or_nan(
                        cols[17], cols[17] == "-999", nm_to_m
                    ),
                    max_high_wind_radius_sw=_apply_or_nan(
                        cols[18], cols[18] == "-999", nm_to_m
                    ),
                    max_high_wind_radius_nw=_apply_or_nan(
                        cols[19], cols[19] == "-999", nm_to_m
                    ),
                    max_sustained_wind_speed_radius=_apply_or_nan(
                        cols[20], cols[20] == "-999", nm_to_m
                    ),
                )
            )
            data_line_count -= 1
    return track_data

import enum
import os
import re
import tempfile
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import xarray as xr

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

_DEFAULT_NAME = "hurdat2"
_ATLANTIC_BASIN_URL = "https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html"
_PACIFIC_BASIN_URL = "https://www.aoml.noaa.gov/hrd/hurdat/hurdat2-nepac.html"

_DEFAULT_FILE_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", _DEFAULT_NAME)
os.makedirs(_DEFAULT_FILE_PATH, exist_ok=True)


class BasinOption(int, enum.Enum):
    ATLANTIC = 1
    PACIFIC = 2
    BOTH = 3


class RecordIdentifier(str, enum.Enum):
    """C – Closest approach to a coast, not followed by a landfall
    G – Genesis
    I – An intensity peak in terms of both pressure and wind
    L – Landfall (center of system crossing a coastline)
    P – Minimum in central pressure
    R – Provides additional detail on the intensity of the cyclone when rapid changes are underway
    S – Change of status of the system
    T – Provides additional detail on the track (position) of the cyclone
    W – Maximum sustained wind speed"""

    CLOSES_TO_COAST = "C"
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
    """TD – Tropical cyclone of tropical depression intensity (< 34 knots)
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
    TY - UNKNOWN found in Northeast Pacific Basin"""

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
    basin: str = field(
        metadata={
            "long_name": "Basin",
            "comments": "basin of origin.",
        }
    )
    cyclone_number: int = field(
        metadata={"long_name": "ATCF cyclone number for associated year."}
    )
    year: int = field(metadata={"long_name": "Year"})
    name: str = field(metadata={"long_name": "Name", "na_values": "UNNAMED"})
    rowsize: int = field(metadata={"long_name": "Number of best track entries"})


@dataclass
class DataLine:
    time: float = field(
        metadata={"comments": "Computed property from YYY-MM-DD HH:MM in UTC"}
    )
    record_identifier: RecordIdentifier = field(
        metadata={"long_name": "Record Idenfier", "comments": RecordIdentifier.__doc__}
    )
    system_status: SystemStatus = field(
        metadata={"long_name": "System Status", "comments": SystemStatus.__doc__}
    )
    lat: float = field(
        metadata={
            "long_name": "Latitude",
            "units": "degrees",
            "°": "Heading rounded to 10^-2 place",
        }
    )
    lon: float = field(
        metadata={
            "long_name": "Longitude",
            "units": "degrees",
            "°": "Heading rounded to 10^-2 place",
        }
    )
    wind_speed: float = field(
        metadata={"long_name": "Maximum Sustained Wind Speed", "units": "knots"}
    )
    pressure: float = field(
        metadata={"long_name": "Minimum Pressure", "units": "millibar"}
    )
    low_wind_radii_ne: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent NE Quadrant",
            "units": "nautical-miles",
        }
    )
    low_wind_radii_se: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent SE Quadrant",
            "units": "nautical-miles",
        }
    )
    low_wind_radii_sw: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent SW Quadrant",
            "units": "nautical-miles",
        }
    )
    low_wind_radii_nw: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent NW Quadrant",
            "units": "nautical-miles",
        }
    )
    med_wind_radii_ne: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent NE Quadrant",
            "units": "nautical-miles",
        }
    )
    med_wind_radii_se: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent SE Quadrant",
            "units": "nautical-miles",
        }
    )
    med_wind_radii_sw: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent SW Quadrant",
            "units": "nautical-miles",
        }
    )
    med_wind_radii_nw: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent NW Quadrant",
            "units": "nautical-miles",
        }
    )
    high_wind_radii_ne: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent NE Quadrant",
            "units": "nautical-miles",
        }
    )
    high_wind_radii_se: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent SE Quadrant",
            "units": "nautical-miles",
        }
    )
    high_wind_radii_sw: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent SW Quadrant",
            "units": "nautical-miles",
        }
    )
    high_wind_radii_nw: float = field(
        metadata={
            "comment": "34 kt Wind Radii Maximum Extent NW Quadrant",
            "units": "nautical-miles",
        }
    )
    max_sustained_radii: float = field(
        metadata={"long_name": "Radius of Maximum Wind", "units": "nautical-miles"}
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

    def get_genesis_date(self) -> float:
        results = list(
            filter(lambda d: d.record_identifier == RecordIdentifier.GENESIS, self.data)
        )
        if len(results) > 0:
            return results[0].time
        return sorted(self.data, key=lambda x: x.time)[0].time

    def to_xarray_dataset(self, id_: int) -> xr.Dataset:
        return xr.Dataset(
            {
                "basin": (["traj"], np.array([self.header.basin])),
                "name": (["traj"], np.array([self.header.name])),
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
                "low_wind_radii_ne": (
                    ["obs"],
                    np.array([line.low_wind_radii_ne for line in self.data]),
                ),
                "low_wind_radii_se": (
                    ["obs"],
                    np.array([line.low_wind_radii_se for line in self.data]),
                ),
                "low_wind_radii_sw": (
                    ["obs"],
                    np.array([line.low_wind_radii_sw for line in self.data]),
                ),
                "low_wind_radii_nw": (
                    ["obs"],
                    np.array([line.low_wind_radii_nw for line in self.data]),
                ),
                "med_wind_radii_ne": (
                    ["obs"],
                    np.array([line.med_wind_radii_ne for line in self.data]),
                ),
                "med_wind_radii_se": (
                    ["obs"],
                    np.array([line.med_wind_radii_se for line in self.data]),
                ),
                "med_wind_radii_sw": (
                    ["obs"],
                    np.array([line.med_wind_radii_sw for line in self.data]),
                ),
                "med_wind_radii_nw": (
                    ["obs"],
                    np.array([line.med_wind_radii_nw for line in self.data]),
                ),
                "high_wind_radii_ne": (
                    ["obs"],
                    np.array([line.high_wind_radii_ne for line in self.data]),
                ),
                "high_wind_radii_se": (
                    ["obs"],
                    np.array([line.high_wind_radii_se for line in self.data]),
                ),
                "high_wind_radii_sw": (
                    ["obs"],
                    np.array([line.high_wind_radii_sw for line in self.data]),
                ),
                "high_wind_radii_nw": (
                    ["obs"],
                    np.array([line.high_wind_radii_nw for line in self.data]),
                ),
            },
            coords={
                "id": (["traj"], np.array([id_])),
                "time": (
                    ["obs"],
                    np.array([line.time for line in self.data], dtype=np.float64),
                ),
            },
        )

    def get_rowsize(self) -> int:
        return len(self.data)


def _map_heading(coordinate: str) -> float:
    heading_map = {"N": 1, "E": 1, "S": -1, "W": -1}
    cardinal_direction = coordinate[-1]
    heading = coordinate[:-1]

    if (sign := heading_map.get(cardinal_direction)) is not None:
        return float(heading) * sign
    raise ValueError(f"Invalid cardinal direction: {cardinal_direction}")


def _get_download_requests(basin: BasinOption):
    download_requests: list[tuple[str, str, None]] = list()

    if basin & BasinOption.ATLANTIC == BasinOption.ATLANTIC:
        file_name = _ATLANTIC_BASIN_URL.split("/")[-1]
        fp = os.path.join(_DEFAULT_FILE_PATH, file_name)
        download_requests.append((_ATLANTIC_BASIN_URL, fp, None))

    if basin & BasinOption.PACIFIC == BasinOption.PACIFIC:
        file_name = _PACIFIC_BASIN_URL.split("/")[-1]
        fp = os.path.join(_DEFAULT_FILE_PATH, file_name)
        download_requests.append((_PACIFIC_BASIN_URL, fp, None))
    return download_requests


def to_raggedarray(basin: BasinOption = BasinOption.BOTH) -> RaggedArray:
    download_requests = _get_download_requests(basin)
    download_with_progress(download_requests)
    track_data = list()

    for _, fp, _ in download_requests:
        track_data.extend(extract_track_data(fp))

    track_data = sorted(track_data, key=lambda td: td.get_genesis_date())
    track_data_map: dict[int, TrackData] = {
        idx: track_data[idx] for idx in range(0, len(track_data))
    }

    metadata_fields = list()
    data_fields = list()

    for f in fields(HeaderLine):
        if f.name != "id":
            metadata_fields.append(f.name)

    for f in fields(DataLine):
        if f.name != "time":
            data_fields.append(f.name)

    ra = RaggedArray.from_items(
        indices=list(track_data_map.keys()),
        name_coords=["id", "time"],
        name_meta=metadata_fields,
        name_data=data_fields,
        name_dims={"traj": "rows", "obs": "obs"},
        rowsize_func=lambda idx: track_data_map[idx].get_rowsize(),
        preprocess_func=lambda idx: track_data_map[idx].to_xarray_dataset(idx),
        attrs_global=TrackData.global_attrs,
        attrs_variables={
            field.name: field.metadata
            for field in fields(HeaderLine) + fields(DataLine)
        },
    )
    return ra


def extract_track_data(datafile_path: str) -> list[TrackData]:
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
                basin=cols[0][:2],
                cyclone_number=int(cols[0][2:4]),
                year=int(cols[0][4:7]),
                name=cols[1],
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
                    ).timestamp(),
                    record_identifier=RecordIdentifier(cols[2]),
                    system_status=SystemStatus(cols[3]),
                    lat=_map_heading(cols[4]),
                    lon=_map_heading(cols[5]),
                    wind_speed=float(cols[6]),
                    pressure=float(cols[7]),
                    low_wind_radii_ne=float(cols[8]),
                    low_wind_radii_se=float(cols[9]),
                    low_wind_radii_sw=float(cols[10]),
                    low_wind_radii_nw=float(cols[11]),
                    med_wind_radii_ne=float(cols[12]),
                    med_wind_radii_se=float(cols[13]),
                    med_wind_radii_sw=float(cols[14]),
                    med_wind_radii_nw=float(cols[15]),
                    high_wind_radii_ne=float(cols[16]),
                    high_wind_radii_se=float(cols[17]),
                    high_wind_radii_sw=float(cols[18]),
                    high_wind_radii_nw=float(cols[19]),
                    max_sustained_radii=float(cols[20]),
                )
            )
            data_line_count -= 1
    return track_data

from __future__ import annotations

import datetime
import gzip
import logging
import math
import os
import tempfile
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, override

from tqdm import tqdm

from clouddrift.adapters.gdp import get_gdp_metadata
from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/pub/pazos/data/shane/sst"
_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdpraw")
_FILNAME_TEMPLATE = "buoydata_{start}_{end}_{suffix}.dat.gz"
os.makedirs(
    _TMP_PATH, exist_ok=True
)  # generate temp directory for hurdat related intermerdiary data
_SECONDS_IN_DAY = 86_400

_RecordKind = Literal["position"] | Literal["sensor"] | Literal["raw"]

_logger = logging.getLogger(__name__)


@dataclass
class DrifterMetadata:
    id: int = field(
        metadata=dict()
    )
    wmoNumber: int = field(
        metadata=dict()
    )
    programNumber: int = field(
        metadata=dict()
    )
    buoyType: str = field(
        metadata=dict()
    )
    startDateTime: datetime.datetime = field(
        metadata=dict()
    )
    startLat: float = field(
        metadata=dict()
    )
    startLon: float = field(
        metadata=dict()
    )
    endDateTime: datetime.datetime = field(
        metadata=dict()
    )
    endLat: float = field(
        metadata=dict()
    )
    endLon: float = field(
        metadata=dict()
    )
    drogueOffDateTime: str = field(
        metadata=dict()
    )
    deathCode: int = field(
        metadata=dict()
    )


@dataclass
class RawDrifterBase:
    id: int = field(
        metadata=dict()
    )

    @classmethod
    @abstractmethod
    def fromcols(cls, cols: list[str]) -> RawDrifterBase:
        raise NotImplementedError()


@dataclass
class RawDrifterPosition:
    positionDateTime: datetime.datetime = field(
        metadata=dict()
    )
    lat: float = field(
        metadata=dict()
    )
    lon: float = field(
        metadata=dict()
    )
    qualityIndex: float


@dataclass
class RawDrifterSensor:
    sensorDateTime: datetime.datetime = field(
        metadata=dict()
    )
    drogue: float = field(
        metadata=dict()
    )
    sst: float = field(
        metadata=dict()
    )
    voltage: float = field(
        metadata=dict()
    )
    sensor4: float = field(
        metadata=dict()
    )
    sensor5: float = field(
        metadata=dict()
    )
    sensor6: float = field(
        metadata=dict()
    )


@dataclass
class RawDrifterPRecord(RawDrifterPosition, RawDrifterBase):
    @override
    @classmethod
    def fromcols(cls, cols: list[str]):
        day = math.floor(float(cols[2]))
        dayratio = float(cols[2]) - day
        return cls(
            id=int(cols[0]),
            positionDateTime=datetime.datetime(
                month=int(cols[1]),
                day=day,
                year=int(cols[3])
            ) + datetime.timedelta(seconds=math.ceil(dayratio * _SECONDS_IN_DAY)),
            lat=float(cols[4]),
            lon=float(cols[5]),
            qualityIndex=float(cols[6]),
        )


@dataclass
class RawDrifterSRecord(RawDrifterSensor, RawDrifterBase):
    @override
    @classmethod
    def fromcols(cls, cols: list[str]):
        day = math.floor(float(cols[2]))
        dayratio = float(cols[2]) - day
        return cls(
            id=int(cols[0]),
            sensorDateTime=datetime.datetime(
                month=int(cols[1]),
                day=day,
                year=int(cols[3])
            ) + datetime.timedelta(seconds=math.ceil(dayratio * _SECONDS_IN_DAY)),
            drogue=float(cols[4]),
            sst=float(cols[5]),
            voltage=float(cols[6]),
            sensor4=float(cols[7]),
            sensor5=float(cols[8]),
            sensor6=float(cols[9]),
        )


@dataclass
class RawDrifterPSRecord(RawDrifterPosition, RawDrifterSensor, RawDrifterBase):
    @override
    @classmethod
    def fromcols(cls, cols: list[str]):
        pos_day = math.floor(float(cols[2]))
        pos_dayratio = float(cols[2]) - pos_day

        sensor_day = math.floor(float(cols[7]))
        sensor_dayratio = float(cols[7]) - sensor_day
        return cls(
            id=int(cols[0]),
            positionDateTime=datetime.datetime(
                month=int(cols[1]),
                day=pos_day,
                year=int(cols[3])
            ) + datetime.timedelta(seconds=math.ceil(pos_dayratio * _SECONDS_IN_DAY)),
            lat=float(cols[4]),
            lon=float(cols[5]),
            sensorDateTime=datetime.datetime(
                month=int(cols[6]),
                day=sensor_day,
                year=int(cols[8])
            ) + datetime.timedelta(seconds=math.ceil(sensor_dayratio * _SECONDS_IN_DAY)),
            drogue=float(cols[9]),
            sst=float(cols[10]),
            voltage=float(cols[11]),
            sensor4=float(cols[12]),
            sensor5=float(cols[13]),
            sensor6=float(cols[14]),
            qualityIndex=float(cols[15]),
        )


def _form_url(url: str, start: str, end: str, suffix: str) -> str:
    filename = _FILNAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
    return f"{url}/{filename}"


def _form_path(path: str, start: str, end: str, suffix: str) -> str:
    filename = _FILNAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
    return os.path.join(path, filename)

def _form_request(start, end, suffix):
    return (_form_url(_DATA_URL, start, end, suffix), _form_path(_TMP_PATH, start, end, suffix))


def _form_record(kind: _RecordKind, cols: list[str]):
    constructor = {
        "position": RawDrifterPRecord.fromcols,
        "sensor": RawDrifterSRecord.fromcols,
        "both": RawDrifterPSRecord.fromcols
    }.get(kind)
    return constructor(cols)


def _get_download_list(kind: _RecordKind) -> list[tuple[str, str]]:
    suffix = {
        "position": "edited_pfiles",
        "sensor": "edited_sfiles",
        "both": "rawfiles",
    }.get(kind)
    batches = [
        (1, 5000),
        (5001, 10_000),
        (10_001, 15_000),
        (15_001, "current")
    ]

    return [ 
        _form_request(start, end, suffix) for start, end in batches
    ]


def _load_gdp_metadata() -> dict:
    gdp_metadata_df = get_gdp_metadata()
    gdp_metadata = dict()

    for rowidx in tqdm(range(len(gdp_metadata_df)), desc="loading drifter metadata into memory"):
        drifter_md_row = gdp_metadata_df.iloc[rowidx]
        drifter_id = drifter_md_row[["ID"]].values[0]
        drifter_md = DrifterMetadata(
            id=int(drifter_id),
            wmoNumber=int(drifter_md_row[["WMO_number"]].values[0]),
            programNumber=int(drifter_md_row[["program_number"]].values[0]),
            buoyType=drifter_md_row[["buoys_type"]].values[0],
            drogueOffDateTime=drifter_md_row[["Drogue_off_date"]].values[0],
            startDateTime=drifter_md_row[["Start_date"]].values[0],
            startLat=float(drifter_md_row[["Start_lat"]].values[0]),
            startLon=float(drifter_md_row[["Start_lon"]].values[0]),
            endDateTime=drifter_md_row[["End_date"]].values[0],
            endLat=float(drifter_md_row[["End_lat"]].values[0]),
            endLon=float(drifter_md_row[["End_lon"]].values[0]),
            deathCode=drifter_md_row[["death_code"]].values[0],
        )
        gdp_metadata[drifter_id] = drifter_md
    return gdp_metadata

def _download(kind: _RecordKind):

    gdp_metadata_table = _load_gdp_metadata()
    requests = _get_download_list(kind)
    download_with_progress(requests)
    destinations = [dst for (_, dst) in requests]
    data_files = list()
    for fp in tqdm(destinations, desc="decompressing raw data files"):
        uncompressed_fp = fp[:-3]
        if os.path.exists(uncompressed_fp):
            data_files.append(uncompressed_fp)
            continue

        with gzip.open(fp, "r") as cf, open(uncompressed_fp, "w") as f:
            while((data := cf.read()) != b""):
                f.write(data.decode("utf-8"))
            data_files.append(uncompressed_fp)
    
    mapped_records = defaultdict[str, list](list)

    trashed_rows = 0
    for fp in data_files:
        with open(fp, "r") as f, tqdm(desc=f"Loading records from ({fp}) into memory") as bar:
            while ((row := f.readline()) != ""):
                cols = row.split()
                id_ = int(cols[0])

                try:
                    record = _form_record(kind, cols)
                    mapped_records[id_].append(record)
                    bar.update()
                except Exception as e:
                    _logger.debug(f"Data file ({fp}) contains invalid row ({row}). Error: ({e})")
                    trashed_rows += 1
    _logger.info(f"trashed {trashed_rows} rows due to an exception during load time")
    print("hodor")


def to_raggedarray(
    kind: _RecordKind = "both",
    tmp_path: str = _TMP_PATH,
) -> RaggedArray:
    _download(kind)
    ...
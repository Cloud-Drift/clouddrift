import os
import tempfile
from dataclasses import dataclass, field
from typing import Literal

from clouddrift.adapters.gdp import get_gdp_metadata
from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray


@dataclass
class DrifterMetadata:
    id: str = field(
        metadata=dict()
    )
    wmoNumber: str = field(
        metadata=dict()
    )
    programNumber: str = field(
        metadata=dict()
    )
    buoyType: str = field(
        metadata=dict()
    )
    startDateTime: str = field(
        metadata=dict()
    )
    startLat: float = field(
        metadata=dict()
    )
    startLon: float = field(
        metadata=dict()
    )
    endDateTime: str = field(
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
    id: str = field(
        metadata=dict()
    )
    positionDateTime: str = field(
        metadata=dict()
    )
    qualityIndex: float


@dataclass
class RawDrifterPosition:
    lat: float = field(
        metadata=dict()
    )
    lon: float = field(
        metadata=dict()
    )


@dataclass
class RawDrifterSensor:
    drogue: str = field(
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
class RawDrifterPFileRecord(RawDrifterPosition, RawDrifterBase):
    ...

class RawDrifterSFileRecord(RawDrifterSensor, RawDrifterBase):
    ...

@dataclass
class RawDrifterPSFileRecord(RawDrifterPosition, RawDrifterSensor, RawDrifterBase):
    ...


_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/pub/pazos/data/shane/sst"
_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdpraw")
_FILNAME_TEMPLATE = "buoydata_{start}_{end}_{suffix}.dat.gz"



def _form_url(url: str, start: str, end: str, suffix: str) -> str:
    filename = _FILNAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
    return f"{url}/{filename}"


def _form_path(path: str, start: str, end: str, suffix: str) -> str:
    filename = _FILNAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
    return os.path.join(_TMP_PATH, filename)


def _get_download_list(kind: Literal["position"] | Literal["sensor"] | Literal["raw"]) -> list[tuple[str, str]]:
    suffix = {
        "position": "edited_pfiles",
        "sensor": "edited_sfiles",
        "raw": "rawfiles",
    }.get(kind)

    return list(
        map(lambda start, end: (_form_url(_DATA_URL, start, end, suffix), _form_path(_TMP_PATH, start, end, suffix), None), [
            (1, 5000),
            (5001, 10_000),
            (10_001, 15_000),
            (15_001, "current")
        ])
    )


def download(kind: Literal["position"] | Literal["sensor"] | Literal["raw"]):
    download_with_progress(_get_download_list(kind))
    gdp_metadata = get_gdp_metadata()

    ...

def to_raggedarray(
    drifter_ids: list[int] | None = None,
    n_random_id: int | None = None,
    tmp_path: str = _TMP_PATH,
) -> RaggedArray:
    ...
from __future__ import annotations

import datetime
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import xarray as xr
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


def _parse_datetime_with_day_ratio(monthSeries: np.ndarray, day_series: np.ndarray, year_series: np.ndarray):
    values = list()
    for month, day_with_ratio, year in zip(monthSeries, day_series, year_series):
        day = day_with_ratio // 1
        dayratio = day_with_ratio - day
        seconds = dayratio * _SECONDS_IN_DAY
        dt_ns = (datetime.datetime(year=int(year), month=int(month), day=int(day)) + datetime.timedelta(seconds=seconds)).timestamp() * 10**9
        values.append(int(dt_ns))
    return np.array(values, dtype="datetime64[ns]")


@dataclass
class ParsingConfiguration:
    cols: list[str]
    coords: list[str]
    col_dtypes: dict[str, np.dtype]
    transform: dict[str, tuple[list[str], Callable[..., np.ndarray]]]

    def get_vars_config_map(self, dim_name: str, df: pd.DataFrame):
        all_config_map = self._get_all_config_map(dim_name, df)
        return {col: all_config_map[col] for col in all_config_map.keys() if col not in self.coords}

    def get_coords_config_map(self, dim_name: str, df: pd.DataFrame):
        all_config_map = self._get_all_config_map(dim_name, df)
        return {col: all_config_map[col] for col in self.coords}

    def _get_all_config_map(self, dim_name: str, df: pd.DataFrame):
        pre_transform = {col: ([dim_name], df[[col]].to_numpy().flatten()) for col in self.cols}
        post_transform = self._apply_transform(pre_transform)
        return post_transform

    def _apply_transform(self, variable_config_map: dict[str, tuple[list[str], pd.DataFrame]]):
        for output_col in self.transform.keys():
            dim, input_cols, func = self.transform[output_col] 
            args = list()
            for col in input_cols:
                _, variable = variable_config_map[col]
                args.append(variable)
            variable_config_map[output_col] = ([dim], func(*args))
            [variable_config_map.pop(col) for col in input_cols]
        return variable_config_map


def _get_parsing_config(kind: _RecordKind) -> ParsingConfiguration:
    return {
        "position": ParsingConfiguration(
            cols=["id", "obsMonth", "obsDay", "obsYear", "lat", "lon", "qualityIndex"],
            col_dtypes={
                "id": np.int64,
                "obsMonth": np.int8,
                "obsDay": np.float16,
                "obsYear": np.int16,
                "lat": np.float32,
                "lon": np.float32,
                "qualityIndex": np.float32
            },
            transform={
                "obsDatetime": ("obs", ["obsMonth", "obsDay", "obsYear"], _parse_datetime_with_day_ratio)
            },
            coords=["id", "obsDatetime"]
        ),
        "sensor": ParsingConfiguration(
            cols=["id", "obsMonth", "obsDay", "obsYear", "drogue", "sst", "voltage", "sensor4", "sensor5", "sensor6"],
            col_dtypes={
                "id": np.int64, 
                "obsMonth": np.int8,
                "obsDay": np.float16,
                "obsYear": np.int16,
                "drogue": np.int16,
                "sst": np.float32,
                "volate": np.float32,
                "sensor4": np.float32,
                "sensor5": np.float32,
                "sensor6": np.float32
            },
            transform={
                "obsDatetime": ("obs", ["obsMonth", "obsDay", "obsYear"], _parse_datetime_with_day_ratio)
            },
            coords=["id", "obsDatetime"]
        ),
        "both": ParsingConfiguration(
            cols=["id", "posObsMonth", "posObsDay", "posObsYear", "lat", "lon", "senObsMonth", "senObsDay", "senObsYear", "drogue", "sst", "voltage", "sensor4", "sensor5", "sensor6", "qualityIndex"],
            col_dtypes={
                "id": np.int64, 
                "posObsMonth": np.int8,
                "posObsDay": np.float16,
                "posObsYear": np.int16,
                "lat": np.float32,
                "lon": np.float32,
                "qualityIndex": np.float32,
                "senObsMonth": np.int8,
                "senObsDay": np.float16,
                "senObsYear": np.int16,
                "drogue": np.int16,
                "sst": np.float32,
                "volate": np.float32,
                "sensor4": np.float32,
                "sensor5": np.float32,
                "sensor6": np.float32,
            },
            transform={
                "posObsDatetime": ("obs", ["posObsMonth", "posObsDay", "posObsYear"], _parse_datetime_with_day_ratio),
                "sensorObsDatetime": ("obs", ["senObsMonth", "senObsDay", "senObsYear"], _parse_datetime_with_day_ratio)
            },
            coords=["id", "posObsDatetime", "sensorObsDatetime"]
        )
    }.get(kind)


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

    requests = list()

    for start, end in batches:
        filename = _FILNAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
        requests.append((f"{_DATA_URL}/{filename}", os.path.join(_TMP_PATH, filename)))
    return requests

def setup_preprocess(md_df: pd.DataFrame, data_df: pd.DataFrame, config: ParsingConfiguration):
    def wrapped_preprocess(id_) -> xr.Dataset:
        traj_md_df = md_df[md_df["ID"] == id_[0]]
        traj_data_df = data_df[data_df["id"] == id_[0]]

        variables = {
            "wmo_number": (["traj"], traj_md_df[["WMO_number"]].values[0]),
            "program_number": (["traj"], traj_md_df[["program_number"]].values[0]),
            "buoys_type": (["traj"], traj_md_df[["buoys_type"]].values[0]),
            "start_date": (["traj"], traj_md_df[["Start_date"]].values[0]),
            "start_lat": (["traj"], traj_md_df[["Start_lat"]].values[0]),
            "start_lon": (["traj"], traj_md_df[["Start_lon"]].values[0]),
            "end_date": (["traj"], traj_md_df[["End_date"]].values[0]),
            "end_lat": (["traj"], traj_md_df[["End_lat"]].values[0]),
            "end_lon": (["traj"], traj_md_df[["End_lon"]].values[0]),
            "drogue_off_date": (["traj"], traj_md_df[["Drogue_off_date"]].values[0]),
            "death_code": (["traj"], traj_md_df[["death_code"]].values[0]),
        }

        data_variables = config.get_vars_config_map("obs", traj_data_df)
        coords = config.get_coords_config_map("obs", traj_data_df)
        row_coord = {
            "id": (["traj"], traj_md_df[["ID"]].values[0]),
        }

        variables.update(data_variables),
        coords.update(row_coord),
        
        dataset = xr.Dataset(
            variables,
            coords=coords
        )
        return dataset
    return wrapped_preprocess


def to_raggedarray(
    kind: _RecordKind = "both",
    tmp_path: str = _TMP_PATH,
) -> RaggedArray:
    gdp_metadata_df = get_gdp_metadata()

    config = _get_parsing_config(kind)
    requests = _get_download_list(kind)
    download_with_progress(requests[:1])
    destinations = [dst for (_, dst) in requests]

    df: pd.DataFrame = None
    for fp in tqdm(destinations[:1], desc="loading data files"):
        before = datetime.datetime.now()

        cur_df = pd.read_csv(
            fp, 
            sep=r"\s+",
            header=None,
            names=config.cols,
            engine="c",
            compression="gzip",
        )

        after = datetime.datetime.now()

        _logger.debug(f"elapsed time to load {len(cur_df)} records :: {after - before}")

        if df is None:
            df = cur_df
        else:
            df = pd.concat([df, cur_df])

    
    ra = RaggedArray.from_files(
        indices=gdp_metadata_df[["ID"]].values,
        preprocess_func=setup_preprocess(gdp_metadata_df, df, config),
        name_coords=config.coords,
        name_meta=[
            "ID",
            "WMO_number",
            "program_number",
            "buoys_type",
            "Start_date",
            "Start_lat",
            "Start_lon",
            "End_date",
            "End_lat",
            "End_lon",
            "Drogue_off_date",
            "death_code",
        ],
        name_data=["lat", "lon", "drogue", "sst", "voltage", "sensor4", "sensor5", "sensor6"],
        name_dims={"traj": "rows", "obs": "obs"},
        rowsize_func=lambda id_: len(df[df["id"] == id_[0]])
    )
    return ra

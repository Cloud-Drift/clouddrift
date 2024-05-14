from __future__ import annotations

import asyncio
import datetime
import logging
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.asyncio import tqdm

from clouddrift.adapters.gdp import get_gdp_metadata
from clouddrift.adapters.utils import download_with_progress
from clouddrift.ragged import subset
from clouddrift.raggedarray import RaggedArray

_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/pub/pazos/data/shane/sst"
_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdpraw")
_FILNAME_TEMPLATE = "buoydata_{start}_{end}_{suffix}.dat.gz"
_SECONDS_IN_DAY = 86_400

_DATA_VARS = [
    "lat",
    "lon",
    "drogue",
    "sst",
    "voltage",
    "sensor4",
    "sensor5",
    "sensor6",
]

_METADATA_VARS = [
    "rowsize",
    "wmo_number",
    "program_number",
    "buoys_type",
    "start_date",
    "start_lat",
    "start_lon",
    "end_date",
    "end_lat",
    "end_lon",
    "drogue_off_date",
    "death_code",
]

_RecordKind = Literal["position"] | Literal["sensor"] | Literal["raw"]

_logger = logging.getLogger(__name__)


def _parse_datetime_with_day_ratio(
    monthSeries: np.ndarray, day_series: np.ndarray, year_series: np.ndarray
):
    values = list()
    for month, day_with_ratio, year in zip(monthSeries, day_series, year_series):
        day = day_with_ratio // 1
        dayratio = day_with_ratio - day
        seconds = dayratio * _SECONDS_IN_DAY
        dt_ns = (
            datetime.datetime(year=int(year), month=int(month), day=int(1))
            + datetime.timedelta(days=int(day), seconds=seconds)
        ).timestamp() * 10**9
        values.append(int(dt_ns))
    return np.array(values, dtype="datetime64[ns]")


@dataclass
class ParsingConfiguration:
    cols: list[str]
    coords: list[str]
    col_dtypes: dict[str, type]
    remove: list[Callable[[pd.DataFrame], pd.DataFrame]]
    transform: dict[str, tuple[str, list[str], Callable[..., pd.DataFrame]]]

    def get_vars_config_map(self, dim_name: str, df: pd.DataFrame):
        all_config_map = self._get_all_config_map(dim_name, df)
        return {
            col: all_config_map[col]
            for col in all_config_map.keys()
            if col not in self.coords
        }

    def get_coords_config_map(self, dim_name: str, df: pd.DataFrame):
        all_config_map = self._get_all_config_map(dim_name, df)
        return {col: all_config_map[col] for col in self.coords}

    def _get_all_config_map(self, dim_name: str, df: pd.DataFrame):
        post_remove = self._apply_remove(df)
        pre_transform = dict()

        for col in self.cols:
            data_array = (
                post_remove[[col]].to_numpy().flatten().astype(self.col_dtypes[col])
            )
            pre_transform[col] = ([dim_name], data_array)
        post_transform = self._apply_transform(pre_transform)
        return post_transform

    def _apply_remove(self, df: pd.DataFrame):
        temp_df = df
        for filter_ in self.remove:
            mask = filter_(temp_df)
            temp_df = temp_df[~mask]
        return temp_df

    def _apply_transform(
        self, variable_config_map: dict[str, tuple[list[str], pd.DataFrame]]
    ):
        for output_col in self.transform.keys():
            dim, input_cols, func = self.transform[output_col]
            args = list()
            for col in input_cols:
                _, variable = variable_config_map[col]
                args.append(variable)
            variable_config_map[output_col] = ([dim], func(*args))
            [variable_config_map.pop(col) for col in input_cols]
        return variable_config_map


def _future_years_mask(df):
    return df["obsYear"] > datetime.datetime.now().year


def _future_years_mask_sen(df):
    return df["posObsYear"] > datetime.datetime.now().year


def _future_years_mask_pos(df):
    return df["senObsYear"] > datetime.datetime.now().year


def _bad_drogue_values_mask(df):
    return df["drogue"].astype(np.str_).str.match(r"(\d+[\.]+){2,}")


def _get_parsing_config(kind: _RecordKind) -> ParsingConfiguration:
    cfg = {
        "position": ParsingConfiguration(
            cols=["id", "obsMonth", "obsDay", "obsYear", "lat", "lon", "qualityIndex"],
            col_dtypes={
                "id": np.int64,
                "obsMonth": np.int8,
                "obsDay": np.float16,
                "obsYear": np.int16,
                "lat": np.float32,
                "lon": np.float32,
                "qualityIndex": np.float32,
            },
            transform={
                "obsDatetime": (
                    "obs",
                    ["obsMonth", "obsDay", "obsYear"],
                    _parse_datetime_with_day_ratio,
                )
            },
            remove=[_future_years_mask, _bad_drogue_values_mask],
            coords=["id", "obsDatetime"],
        ),
        "sensor": ParsingConfiguration(
            cols=[
                "id",
                "obsMonth",
                "obsDay",
                "obsYear",
                "drogue",
                "sst",
                "voltage",
                "sensor4",
                "sensor5",
                "sensor6",
            ],
            col_dtypes={
                "id": np.int64,
                "obsMonth": np.int8,
                "obsDay": np.float16,
                "obsYear": np.int16,
                "drogue": np.float16,
                "sst": np.float32,
                "voltage": np.float32,
                "sensor4": np.float32,
                "sensor5": np.float32,
                "sensor6": np.float32,
            },
            transform={
                "obsDatetime": (
                    "obs",
                    ["obsMonth", "obsDay", "obsYear"],
                    _parse_datetime_with_day_ratio,
                )
            },
            remove=[_future_years_mask, _bad_drogue_values_mask],
            coords=["id", "obsDatetime"],
        ),
        "both": ParsingConfiguration(
            cols=[
                "id",
                "posObsMonth",
                "posObsDay",
                "posObsYear",
                "lat",
                "lon",
                "senObsMonth",
                "senObsDay",
                "senObsYear",
                "drogue",
                "sst",
                "voltage",
                "sensor4",
                "sensor5",
                "sensor6",
                "qualityIndex",
            ],
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
                "drogue": np.float16,
                "sst": np.float32,
                "voltage": np.float32,
                "sensor4": np.float32,
                "sensor5": np.float32,
                "sensor6": np.float32,
            },
            transform={
                "posObsDatetime": (
                    "obs",
                    ["posObsMonth", "posObsDay", "posObsYear"],
                    _parse_datetime_with_day_ratio,
                ),
                "sensorObsDatetime": (
                    "obs",
                    ["senObsMonth", "senObsDay", "senObsYear"],
                    _parse_datetime_with_day_ratio,
                ),
            },
            remove=[
                _future_years_mask_pos,
                _future_years_mask_sen,
                _bad_drogue_values_mask,
            ],
            coords=["id", "posObsDatetime", "sensorObsDatetime"],
        ),
    }.get(kind)

    if cfg is None:
        raise ValueError(f"The {kind} kind doesn't have an associated parsing configuration")
    return cfg


def _get_download_list(tmp_path: str, kind: _RecordKind) -> list[tuple[str, str]]:
    suffix = {
        "position": "edited_pfiles",
        "sensor": "edited_sfiles",
        "both": "rawfiles",
    }.get(kind)
    batches = [(1, 5000), (5001, 10_000), (10_001, 15_000), (15_001, "current")]

    requests = list()

    for start, end in batches:
        filename = _FILNAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
        requests.append((f"{_DATA_URL}/{filename}", os.path.join(tmp_path, filename)))
    return requests


def rowsize(id_, **kwargs) -> int:
    df: pd.DataFrame | None = kwargs.get("data_df")
    config: ParsingConfiguration | None = kwargs.get("config")

    if df is None or config is None:
        raise KeyError("Missing `data_df` or `config`, please pass them into the `from_files` method")

    traj_data_df = df[df["id"] == id_]
    coords = config.get_coords_config_map("obs", traj_data_df)
    _, var = coords[
        list(coords.keys())[0]
    ]  # any of the coords will work, to determine the rowsize
    return len(var)


def preprocess(id_, **kwargs) -> xr.Dataset:
    md_df: pd.DataFrame | None = kwargs.get("md_df")
    data_df: pd.DataFrame | None = kwargs.get("data_df")
    config: ParsingConfiguration | None = kwargs.get("config")

    if md_df is None or data_df is None or config is None:
        raise KeyError("Missing `md_df` or `data_df` or `config`, please pass them into the `from_files` method")

    traj_md_df = md_df[md_df["ID"] == id_]
    traj_data_df = data_df[data_df["id"] == id_]

    coords = config.get_coords_config_map("obs", traj_data_df)
    _, var = coords[
        list(coords.keys())[0]
    ]  # any of the coords will work, to determine the rowsize
    rowsize = len(var)

    variables = {
        "rowsize": (["traj"], np.array([rowsize], dtype=np.int64)),
        "wmo_number": (
            ["traj"],
            traj_md_df[["WMO_number"]].values[0].astype(np.int64),
        ),
        "program_number": (
            ["traj"],
            traj_md_df[["program_number"]].values[0].astype(np.int64),
        ),
        "buoys_type": (
            ["traj"],
            traj_md_df[["buoys_type"]].values[0].astype(np.str_),
        ),
        "start_date": (
            ["traj"],
            traj_md_df[["Start_date"]].values[0].astype(np.datetime64),
        ),
        "start_lat": (
            ["traj"],
            traj_md_df[["Start_lat"]].values[0].astype(np.float64),
        ),
        "start_lon": (
            ["traj"],
            traj_md_df[["Start_lon"]].values[0].astype(np.float64),
        ),
        "end_date": (
            ["traj"],
            traj_md_df[["End_date"]].values[0].astype(np.datetime64),
        ),
        "end_lat": (["traj"], traj_md_df[["End_lat"]].values[0].astype(np.float64)),
        "end_lon": (["traj"], traj_md_df[["End_lon"]].values[0].astype(np.float64)),
        "drogue_off_date": (
            ["traj"],
            traj_md_df[["Drogue_off_date"]].values[0].astype(np.datetime64),
        ),
        "death_code": (
            ["traj"],
            traj_md_df[["death_code"]].values[0].astype(np.int64),
        ),
    }

    data_variables = config.get_vars_config_map("obs", traj_data_df)
    coords = config.get_coords_config_map("obs", traj_data_df)
    row_coord = {
        "id": (["traj"], traj_md_df[["ID"]].values[0].astype(np.int64)),
    }

    variables.update(data_variables)
    coords.update(row_coord)

    dataset = xr.Dataset(variables, coords=coords)
    return dataset


def _process_chunk(
    df_chunk: pd.DataFrame,
    chunk_id: str,
    gdp_metadata_df: pd.DataFrame,
    config: ParsingConfiguration,
):
    filename = f"{chunk_id}.zarr"
    zarr_path = os.path.join(_TMP_PATH, "chunks", filename)

    # remove the current zar archive if it exists
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    ids_in_data = np.unique(df_chunk[["id"]].values)
    ids_with_md = np.intersect1d(ids_in_data, gdp_metadata_df[["ID"]].values)

    if len(ids_in_data) > len(ids_with_md):
        warnings.warn(
            "Data has drifter ids not found in the metadata table. "
            + "Please inspect the following ids"
        )

    ra = RaggedArray.from_files(
        indices=ids_with_md,
        preprocess_func=preprocess,
        rowsize_func=rowsize,
        name_coords=config.coords,
        name_meta=_METADATA_VARS,
        name_data=_DATA_VARS,
        name_dims={"traj": "rows", "obs": "obs"},
        md_df=gdp_metadata_df,
        data_df=df_chunk,
        config=config,
        tqdm=dict(disable=True),
    )
    ra.to_xarray().to_zarr(zarr_path)
    return zarr_path, ids_with_md


def _combine_chunked_drifter_datasets(datasets: list[xr.Dataset]):
    new_rowsize = sum([ds.rowsize.values[0] for ds in datasets])
    traj_dataset = xr.concat(
        datasets, dim="obs", coords="minimal", data_vars=_DATA_VARS, compat="override"
    )
    traj_dataset["rowsize"] = xr.DataArray(
        np.array([new_rowsize], dtype=np.int64), coords=traj_dataset["rowsize"].coords
    )
    return traj_dataset


async def _parallel_get(
    sources: list[str],
    gdp_metadata_df: pd.DataFrame,
    config: ParsingConfiguration,
    chunk_size: int,
) -> list[xr.Dataset]:
    max_workers = (os.cpu_count() or 0) // 2
    with ProcessPoolExecutor(max_workers=max_workers) as ppe:
        drifter_chunked_datasets: dict[int, list[xr.Dataset]] = defaultdict(list)
        for fp in tqdm(
            sources,
            desc="Loading files",
            unit="file",
            ncols=80,
            total=len(sources),
            position=0,
        ):
            filename = fp.split(os.path.sep)[-1]
            # it has two extensions since its a tarball (tar) compressed (gzip)
            filename_minus_ext = filename[:-7]
            file_chunks = pd.read_csv(
                fp,
                sep=r"\s+",
                header=None,
                names=config.cols,
                engine="c",
                compression="gzip",
                chunksize=chunk_size,
            )

            joblist = list[Future]()
            for idx, chunk in enumerate(file_chunks):
                ajob = ppe.submit(
                    _process_chunk,
                    chunk,
                    f"{filename_minus_ext}-{idx}",
                    gdp_metadata_df,
                    config,
                )
                joblist.append(ajob)

            bar = tqdm(
                desc="Processing file chunks",
                unit="chunk",
                ncols=80,
                total=len(joblist),
                position=1,
            )

            bad_chunks = 0
            for ajob in as_completed(joblist):
                if ajob.exception() is not None:
                    bad_chunks += 1
                    continue

                fp, ids = ajob.result()
                for id_ in ids:
                    f_ds = xr.open_dataset(fp, engine="zarr")
                    id_f_ds = subset(f_ds, dict(id=id_), row_dim_name="traj")
                    drifter_chunked_datasets[id_].append(id_f_ds)
                bar.update()
            _logger.info(
                f"{bad_chunks} (chunksize: {chunk_size}) dropped."
                + " Note dropped chunks could have been empty"
            )

        joblist = list[Future]()
        for id_ in drifter_chunked_datasets.keys():
            datasets = drifter_chunked_datasets[id_]

            ajob = ppe.submit(_combine_chunked_drifter_datasets, *[datasets])
            joblist.append(ajob)

        bar.close()
        bar = tqdm(
            desc="merging drifter chunks",
            unit="drifter",
            ncols=80,
            total=len(drifter_chunked_datasets.keys()),
            position=2,
        )

        drifter_datasets = list[xr.Dataset]()
        for ajob in as_completed(joblist):
            dataset = ajob.result()
            drifter_datasets.append(dataset)
            bar.update()
        bar.close()
        return drifter_datasets


def get_dataset(
    kind: _RecordKind = "raw",
    tmp_path: str = _TMP_PATH,
    max: int | None = None,
    chunk_size: int = 100_000,
) -> xr.Dataset:
    config = _get_parsing_config(kind)
    requests = _get_download_list(tmp_path, kind)
    destinations = [dst for (_, dst) in requests]

    if max:
        requests = requests[:max]
        destinations = destinations[:max]

    download_with_progress(requests)
    gdp_metadata_df = get_gdp_metadata()

    os.makedirs(tmp_path, exist_ok=True)
    drifter_datasets = asyncio.run(
        _parallel_get(destinations, gdp_metadata_df, config, chunk_size)
    )
    obs_ds = xr.concat(
        [ds.drop_dims("traj") for ds in drifter_datasets],
        dim="obs",
        data_vars=_DATA_VARS,
    )
    traj_ds = xr.concat(
        [ds.drop_dims("obs") for ds in drifter_datasets],
        dim="traj",
        data_vars=_METADATA_VARS,
    )

    agg_ds = xr.merge([obs_ds, traj_ds])

    agg_path = os.path.join(_TMP_PATH, f"gdpraw_{kind}_aggregate.zarr")
    agg_ds.to_zarr(agg_path)
    return agg_ds

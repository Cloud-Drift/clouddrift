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
_FILENAME_TEMPLATE = "buoydata_{start}_{end}_{suffix}.dat.gz"
_SECONDS_IN_DAY = 86_400

_COORDS = [
    "id",
    "obs_index"
]

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


@dataclass
class ParsingConfiguration:
    cols: list[str]
    col_dtypes: dict[str, type]
    remove: list[Callable[[pd.DataFrame], pd.DataFrame]]
    transform: dict[str, tuple[list[str], Callable[..., pd.DataFrame]]]

    def apply_remove(self, df: pd.DataFrame):
        temp_df = df
        for filter_ in self.remove:
            mask = filter_(temp_df)
            temp_df = temp_df[~mask]
        return temp_df

    def apply_transform(self, df: pd.DataFrame):
        tmp_df = df
        for output_col in self.transform.keys():
            input_cols, func = self.transform[output_col]
            args = list()
            for col in input_cols:
                arg = df[[col]].values.flatten()
                args.append(arg)
            tmp_df = tmp_df.assign(**{output_col: func(*args)})
            tmp_df = tmp_df.drop(input_cols, axis=1)
        return tmp_df


def _parse_datetime_with_day_ratio(
    month_series: np.ndarray, day_series: np.ndarray, year_series: np.ndarray
):
    values = list()
    for month, day_with_ratio, year in zip(month_series, day_series, year_series):
        day = day_with_ratio // 1
        dayratio = day_with_ratio - day
        seconds = dayratio * _SECONDS_IN_DAY
        dt_ns = (
            datetime.datetime(year=int(year), month=int(month), day=int(1))
            + datetime.timedelta(days=int(day), seconds=seconds)
        ).timestamp() * 10**9
        values.append(int(dt_ns))
    return np.array(values, dtype="datetime64[ns]")


def _bad_years_mask_sen(df):
    return (df["posObsYear"] > datetime.datetime.now().year) | (df["posObsYear"] < 0)


def _bad_years_mask_pos(df):
    return (df["senObsYear"] > datetime.datetime.now().year) | (df["senObsYear"] < 0)


def _bad_drogue_values_mask(df):
    return df["drogue"].astype(np.str_).str.match(r"(\d+[\.]+){2,}")


def _get_parsing_config(kind: _RecordKind) -> ParsingConfiguration:
    cfg = {
        "raw": ParsingConfiguration(
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
                "posObsDay": np.float64,
                "posObsYear": np.int16,
                "lat": np.float32,
                "lon": np.float32,
                "qualityIndex": np.float32,
                "senObsMonth": np.int8,
                "senObsDay": np.float64,
                "senObsYear": np.int16,
                "drogue": np.float32,
                "sst": np.float32,
                "voltage": np.float32,
                "sensor4": np.float32,
                "sensor5": np.float32,
                "sensor6": np.float32,
            },
            transform={
                "posObsDatetime": (
                    ["posObsMonth", "posObsDay", "posObsYear"],
                    _parse_datetime_with_day_ratio,
                ),
                "sensorObsDatetime": (
                    ["senObsMonth", "senObsDay", "senObsYear"],
                    _parse_datetime_with_day_ratio,
                ),
            },
            remove=[
                _bad_years_mask_pos,
                _bad_years_mask_sen,
                _bad_drogue_values_mask,
            ]
        ),
    }.get(kind)

    if cfg is None:
        raise ValueError(
            f"The {kind} kind doesn't have an associated parsing configuration"
        )
    return cfg


def _get_download_list(tmp_path: str, kind: _RecordKind) -> list[tuple[str, str]]:
    suffix = {
        "position": "edited_pfiles",
        "sensor": "edited_sfiles",
        "raw": "rawfiles",
    }.get(kind)
    batches = [(1, 5000), (5001, 10_000), (10_001, 15_000), (15_001, "current")]

    requests = list()

    for start, end in batches:
        filename = _FILENAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
        requests.append((f"{_DATA_URL}/{filename}", os.path.join(tmp_path, filename)))
    return requests


def rowsize(id_, **kwargs) -> int:
    df: pd.DataFrame | None = kwargs.get("data_df")
    config: ParsingConfiguration | None = kwargs.get("config")

    if df is None or config is None:
        raise KeyError(
            "Missing `data_df` or `config`, please pass them into the `from_files` method"
        )

    traj_data_df = df[df["id"] == id_]
    return len(traj_data_df)


def preprocess(id_, **kwargs) -> xr.Dataset:
    md_df: pd.DataFrame | None = kwargs.get("md_df")
    data_df: pd.DataFrame | None = kwargs.get("data_df")
    config: ParsingConfiguration | None = kwargs.get("config")

    if md_df is None or data_df is None or config is None:
        raise KeyError(
            "Missing `md_df` or `data_df` or `config`, please pass them into the `from_files` method"
        )

    traj_md_df = md_df[md_df["ID"] == id_]
    traj_data_df = data_df[data_df["id"] == id_]
    rowsize = len(traj_data_df)

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

    data_var_names = _DATA_VARS + list(config.transform.keys())
    data_vars = dict()
    for var_name in data_var_names:
        type_  = config.col_dtypes.get(var_name)
        data_vars[var_name] = (["obs"], traj_data_df[[var_name]].values.flatten().astype(type_))

    coords = {
        "id": (["traj"], traj_md_df[["ID"]].values[0].astype(np.int64)),
        "obs_index": (["obs"], traj_data_df[["obs_index"]].values.flatten().astype(np.int32)),
    }

    variables.update(data_vars)

    dataset = xr.Dataset(variables, coords=coords)
    return dataset


def _process_chunk(
    df_chunk: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    gdp_metadata_df: pd.DataFrame,
    config: ParsingConfiguration,
):
    df_chunk = df_chunk.assign(obs_index=range(start_idx, end_idx))
    df_chunk = config.apply_remove(df_chunk)
    df_chunk = df_chunk.astype(config.col_dtypes)
    df_chunk = config.apply_transform(df_chunk)

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
        name_coords=_COORDS,
        name_meta=_METADATA_VARS,
        name_data=_DATA_VARS,
        name_dims={"traj": "rows", "obs": "obs"},
        md_df=gdp_metadata_df,
        data_df=df_chunk,
        config=config,
        tqdm=dict(disable=True),
    )
    ds = ra.to_xarray()

    drifter_ds_map: dict[int, xr.Dataset] = defaultdict(list)
    for id_ in ids_with_md:
        id_f_ds = subset(ds, dict(id=id_), row_dim_name="traj")
        drifter_ds_map[id_] = id_f_ds
    return drifter_ds_map


def _combine_chunked_drifter_datasets(datasets: list[xr.Dataset]):
    """When combining drifter chunks, sort variables using the sort key associated to the dimension.
    A sort key is generated per coordinate and is associated its last dimension.
    """
    traj_dataset = xr.concat(
        datasets, dim="obs", coords="minimal", data_vars=_DATA_VARS, compat="override"
    )

    new_rowsize = sum([ds.rowsize.values[0] for ds in datasets])
    traj_dataset["rowsize"] = xr.DataArray(
        np.array([new_rowsize], dtype=np.int64), coords=traj_dataset["rowsize"].coords
    )

    sort_coord = traj_dataset.coords["obs_index"]
    vals: np.ndarray = sort_coord.data
    sort_coord_dim = sort_coord.dims[-1]
    sort_key = vals.argsort()

    for coord_name in _COORDS:
        coord = traj_dataset.coords[coord_name]
        dim = coord.dims[-1]

        if dim == sort_coord_dim:
            sorted_coord = coord.isel({dim: sort_key})
            traj_dataset.coords[coord_name] = sorted_coord
    
    for varname in _DATA_VARS:
        var = traj_dataset[varname]
        dim = var.dims[-1]
        sorted_var = var.isel({dim: sort_key})
        traj_dataset[varname] = sorted_var

    return traj_dataset


async def _parallel_get(
    sources: list[str],
    gdp_metadata_df: pd.DataFrame,
    config: ParsingConfiguration,
    chunk_size: int,
    tmp_path: str,
) -> list[xr.Dataset]:
    max_workers = (os.cpu_count() or 0) // 2
    with ProcessPoolExecutor(max_workers=max_workers) as ppe:
        drifter_chunked_datasets: dict[int, list[xr.Dataset]] = defaultdict(list)
        start_idx = 0
        for fp in tqdm(
            sources,
            desc="Loading files",
            unit="file",
            ncols=80,
            total=len(sources),
            position=0,
        ):
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
            jobmap = dict[Future, pd.DataFrame]()
            for idx, chunk in enumerate(file_chunks):
                ajob = ppe.submit(
                    _process_chunk,
                    chunk,
                    start_idx,
                    start_idx + len(chunk),
                    gdp_metadata_df,
                    config,
                )
                start_idx += len(chunk)
                jobmap[ajob] = chunk
                joblist.append(ajob)

            bar = tqdm(
                desc="Processing file chunks",
                unit="chunk",
                ncols=80,
                total=len(joblist),
                position=1,
            )

            for ajob in as_completed(jobmap.keys()):
                if ajob.exception() is not None:
                    chunk = jobmap[ajob]
                    _logger.warn(f"bad chunk detected, exception: {ajob.exception()}, ignoring {len(chunk)} rows")
                    raise ajob.exception()

                job_drifter_ds_map: dict[int, xr.Dataset] = ajob.result()
                for id_ in job_drifter_ds_map.keys():
                    drifter_ds = job_drifter_ds_map[id_]
                    drifter_chunked_datasets[id_].append(drifter_ds)
                bar.update()

        jobmap = dict[Future, int]()
        for id_ in drifter_chunked_datasets.keys():
            datasets = drifter_chunked_datasets[id_]

            ajob = ppe.submit(_combine_chunked_drifter_datasets, datasets)
            jobmap[ajob] = id_

        bar.close()
        bar = tqdm(
            desc="merging drifter chunks",
            unit="drifter",
            ncols=80,
            total=len(drifter_chunked_datasets.keys()),
            position=2,
        )

        os.makedirs(os.path.join(tmp_path, "drifters"), exist_ok=True)

        drifter_datasets = list[xr.Dataset]()
        for ajob in as_completed(jobmap.keys()):
            id_ = jobmap[ajob]
            zarr_path = os.path.join(tmp_path, "drifters", f"drifter-{id_}.zarr")

            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)

            dataset: xr.Dataset = ajob.result()
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

    os.makedirs(tmp_path, exist_ok=True)

    if max:
        requests = requests[:max]
        destinations = destinations[:max]

    download_with_progress(requests)
    gdp_metadata_df = get_gdp_metadata(tmp_path)

    drifter_datasets = asyncio.run(
        _parallel_get(destinations, gdp_metadata_df, config, chunk_size, tmp_path)
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

    zarr_path = os.path.join(tmp_path, f"gdpraw_{kind}_aggregate.zarr")

    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    agg_ds.to_zarr(zarr_path)
    return agg_ds

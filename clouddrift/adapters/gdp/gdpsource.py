from __future__ import annotations

import asyncio
import datetime
import logging
import os
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.asyncio import tqdm

from clouddrift.adapters.gdp import get_gdp_metadata
from clouddrift.adapters.utils import download_with_progress
from clouddrift.ragged import subset
from clouddrift.raggedarray import RaggedArray

_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/pub/pazos/data/shane/sst"
_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdpsource")
_FILENAME_TEMPLATE = "buoydata_{start}_{end}_{suffix}.dat.gz"
_SECONDS_IN_DAY = 86_400

_COORDS = ["id", "obs_index"]

_DATA_VARS = [
    "latitude",
    "longitude",
    "position_datetime",
    "sensor_datetime",
    "drogue",
    "sst",
    "voltage",
    "sensor4",
    "sensor5",
    "sensor6",
    "qualityIndex",
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

_VARS_FILL_MAP: dict = {
    "wmo_number": -999,
    "program_number": -999,
    "buoys_type": "N/A",
    "start_date": np.datetime64("1970-01-01 00:00:00"),
    "start_lat": -999,
    "start_lon": -999,
    "end_date": np.datetime64("1970-01-01 00:00:00"),
    "end_lat": -999,
    "end_lon": -999,
    "drogue_off_date": np.datetime64("1970-01-01 00:00:00"),
    "death_code": -999,
}

_VAR_DTYPES: dict = {
    "rowsize": np.int64,
    "wmo_number": np.int64,
    "program_number": np.int64,
    "buoys_type": np.str_,
    "start_date": np.dtype("datetime64[ns]"),
    "start_lat": np.float64,
    "start_lon": np.float64,
    "end_date": np.dtype("datetime64[ns]"),
    "end_lat": np.float64,
    "end_lon": np.float64,
    "drogue_off_date": np.dtype("datetime64[ns]"),
    "death_code": np.int64,
}


_INPUT_COLS = [
    "id",
    "posObsMonth",
    "posObsDay",
    "posObsYear",
    "latitude",
    "longitude",
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
]


_INPUT_COLS_DTYPES = {
    "id": np.int64,
    "posObsMonth": np.int8,
    "posObsDay": np.float64,
    "posObsYear": np.int16,
    "latitude": np.float32,
    "longitude": np.float32,
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
}


VARS_ATTRS: dict = {
    "id": {"long_name": "Global Drifter Program Buoy ID", "units": "-"},
    "rowsize": {
        "long_name": "Number of observations per trajectory",
        "sample_dimension": "obs",
        "units": "-",
    },
    "wmo_number": {
        "long_name": "World Meteorological Organization buoy identification number",
        "units": "-",
    },
    "program_number": {
        "long_name": "Current Program",
        "units": "-",
    },
    "buoys_type": {
        "long_name": "Buoy type (see https://www.aoml.noaa.gov/phod/dac/dirall.html)",
        "units": "-",
    },
    "start_date": {
        "long_name": "First good date and time derived by DAC quality control",
    },
    "start_lon": {
        "long_name": "First good longitude derived by DAC quality control",
        "units": "degrees_east",
    },
    "start_lat": {
        "long_name": "First good latitude derived by DAC quality control",
        "units": "degrees_north",
    },
    "end_date": {
        "long_name": "Last good date and time derived by DAC quality control",
    },
    "end_lon": {
        "long_name": "Last good longitude derived by DAC quality control",
        "units": "degrees_east",
    },
    "end_lat": {
        "long_name": "Last good latitude derived by DAC quality control",
        "units": "degrees_north",
    },
    "drogue_off_date": {
        "long_name": "Date and time of drogue loss",
    },
    "death_code": {
        "long_name": "Type of death",
        "units": "-",
        "comments": "0 (buoy still alive), 1 (buoy ran aground), 2 (picked up by vessel), 3 (stop transmitting), 4 (sporadic transmissions), 5 (bad batteries), 6 (inactive status)",
    },
    "position_datetime": {
        "comments": "Position datetime derived from the year, month, day and time (represented as a ratio of a day) columns found in the source dataset that represent when the position of the drifter was measured. This value is only different from the sensor_datetime when the position of the drifter was determined onboard the Argos satellites using the doppler shift.",
    },
    "sensor_datetime": {
        "comments": "Sensor datetime derived from the year, month, day and time (represented as a ratio of a day) columns found in the source dataset that represent when the sensor (like temp) data is recorded",
    },
    "longitude": {"long_name": "Longitude", "units": "degrees_east"},
    "latitude": {"long_name": "Latitude", "units": "degrees_north"},
    "drogue": {
        "long_name": "Drogue",
        "units": "-",
        "comments": "Values returned by drogue sensor",
    },
    "sst": {
        "long_name": "sea water temperature",
        "units": "degree_Celsius",
        "comments": "sea water temperature from drifting buoy measurements",
    },
    "voltage": {
        "long_name": "Voltage",
        "units": "V",
    },
    "sensor4": {
        "long_name": "Sensor 4",
        "units": "-",
        "comments": "placeholders for additional sensors such as barometers, salinity etc.",
    },
    "sensor5": {
        "long_name": "Sensor 5",
        "units": "-",
        "comments": "placeholders for additional sensors such as barometers, salinity etc.",
    },
    "sensor6": {
        "long_name": "Sensor 6",
        "units": "-",
        "comments": "placeholders for additional sensors such as barometers, salinity etc.",
    },
    "qualityIndex": {
        "long_name": "Quality Index",
        "units": "-",
    },
}

ATTRS = {
    "title": "Global Drifter Program source drifter dataset",
    "Conventions": "CF-1.6",
    "date_created": datetime.datetime.now().isoformat(),
    "publisher_name": "GDP Drifter DAC",
    "publisher_email": "aoml.dftr@noaa.gov",
    "publisher_url": "https://www.aoml.noaa.gov/phod/gdp",
    "license": "freely available",
    "processing_level": "source files",
    "metadata_link": "https://www.aoml.noaa.gov/phod/dac/dirall.html",
    "contributor_name": "NOAA Global Drifter Program",
    "contributor_role": "Data Acquisition Center",
    "institution": "NOAA Atlantic Oceanographic and Meteorological Laboratory",
    "summary": "Global Drifter Program source (raw) data",
}

_logger = logging.getLogger(__name__)


def _get_download_list(tmp_path: str) -> list[tuple[str, str]]:
    suffix = "rawfiles"
    batches = [(1, 5000), (5001, 10_000), (10_001, 15_000), (15_001, "current")]

    requests = list()

    for start, end in batches:
        filename = _FILENAME_TEMPLATE.format(start=start, end=end, suffix=suffix)
        requests.append((f"{_DATA_URL}/{filename}", os.path.join(tmp_path, filename)))
    return requests


def _rowsize(id_, **kwargs) -> int:
    df: pd.DataFrame | None = kwargs.get("data_df")

    if df is None:
        raise KeyError(
            "Missing `data_df`, please pass them into the `from_files` method"
        )

    traj_data_df = df[df["id"] == id_]
    return len(traj_data_df)


def _preprocess(id_, **kwargs) -> xr.Dataset:
    md_df: pd.DataFrame | None = kwargs.get("md_df")
    data_df: pd.DataFrame | None = kwargs.get("data_df")
    use_fill_values: bool = kwargs.get("use_fill_values", False)

    if md_df is None or data_df is None:
        raise KeyError(
            "Missing `md_df` or `data_df`, pass them into the `from_files` method"
        )

    traj_md_df = md_df[md_df["ID"] == id_]
    traj_data_df = data_df[data_df["id"] == id_]
    rowsize = len(traj_data_df)

    md_variables = {
        "rowsize": np.array([rowsize], dtype=np.int64),
    }

    if use_fill_values and len(traj_md_df) == 0:
        for md_var in _METADATA_VARS:
            if md_var == "rowsize":
                continue

            fill_val = _VARS_FILL_MAP.get(md_var)
            if fill_val is None:
                raise ValueError(
                    f"Fill value missing for metadata variable: `{md_var}`"
                )

            md_variables.update({md_var: np.array([fill_val])})
    else:
        md_variables.update(
            {
                "wmo_number": traj_md_df[["WMO_number"]].values.flatten(),
                "program_number": traj_md_df[["program_number"]].values.flatten(),
                "buoys_type": traj_md_df[["buoys_type"]].values.flatten(),
                "start_date": traj_md_df[["Start_date"]].values.flatten(),
                "start_lat": traj_md_df[["Start_lat"]].values.flatten(),
                "start_lon": traj_md_df[["Start_lon"]].values.flatten(),
                "end_date": traj_md_df[["End_date"]].values.flatten(),
                "end_lat": traj_md_df[["End_lat"]].values.flatten(),
                "end_lon": traj_md_df[["End_lon"]].values.flatten(),
                "drogue_off_date": traj_md_df[["Drogue_off_date"]].values.flatten(),
                "death_code": traj_md_df[["death_code"]].values.flatten(),
            }
        )

    for md_var in _METADATA_VARS:
        var = md_variables.get(md_var)

        if var is None:
            raise ValueError(f"Metadata variable `{md_var}` cannot be found")

        md_variables.update({md_var: var.astype(_VAR_DTYPES.get(md_var))})

    variables = {k: (["traj"], md_variables[k]) for k in md_variables}
    data_vars = {
        var: (["obs"], traj_data_df[[var]].values.flatten()) for var in _DATA_VARS
    }

    coords = {
        "id": (["traj"], np.array([id_]).astype(np.int64)),
        "obs_index": (
            ["obs"],
            traj_data_df[["obs_index"]].values.flatten().astype(np.int32),
        ),
    }

    variables.update(data_vars)
    dataset = xr.Dataset(variables, coords=coords)
    return dataset


def _apply_remove(df: pd.DataFrame, filters: list[Callable]) -> pd.DataFrame:
    temp_df = df
    for filter_ in filters:
        mask = filter_(temp_df)
        temp_df = temp_df[~mask]
    return temp_df


def _apply_transform(
    df: pd.DataFrame,
    transforms: dict[str, tuple[list[str], Callable]],
) -> pd.DataFrame:
    tmp_df = df
    for output_col in transforms.keys():
        input_cols, func = transforms[output_col]
        args = list()
        for col in input_cols:
            arg = df[[col]].values.flatten()
            args.append(arg)
        tmp_df = tmp_df.assign(**{output_col: func(*args)})
        tmp_df = tmp_df.drop(input_cols, axis=1)
    return tmp_df


def _parse_datetime_with_day_ratio(
    month_series: np.ndarray, day_series: np.ndarray, year_series: np.ndarray
) -> np.ndarray:
    values = list()
    for month, day_with_ratio, year in zip(month_series, day_series, year_series):
        day = day_with_ratio // 1
        dayratio = day_with_ratio - day
        seconds = dayratio * _SECONDS_IN_DAY
        dt_ns = (
            datetime.datetime(
                year=int(year),
                month=int(month),
                day=int(day),
                tzinfo=datetime.timezone.utc,
            )
            + datetime.timedelta(seconds=seconds)
        ).timestamp() * 10**9
        values.append(int(dt_ns))
    return np.array(values).astype("datetime64[ns]")


def _process_chunk(
    df_chunk: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    gdp_metadata_df: pd.DataFrame,
    use_fill_values: bool,
) -> dict[int, xr.Dataset]:
    """Process each dataframe chunk. Return a dictionary mapping each drifter to a unique xarray Dataset."""

    # Transform the initial dataframe filtering out rows with really anomolous values
    # examples include: years in the future, years way in the past before GDP program, etc...
    preremove_df_chunk = df_chunk.assign(obs_index=np.arange(start_idx, end_idx))
    df_chunk = _apply_remove(
        preremove_df_chunk,
        filters=[
            # Filter out year values that are in the future or predating the GDP program
            lambda df: (df["posObsYear"] > datetime.datetime.now().year)
            | (df["posObsYear"] < 0),
            lambda df: (df["senObsYear"] > datetime.datetime.now().year)
            | (df["senObsYear"] < 0),
            # Filter out month values that contain non-numeric characters
            lambda df: df["senObsMonth"].astype(np.str_).str.contains(r"[\D]"),
            lambda df: df["posObsMonth"].astype(np.str_).str.contains(r"[\D]"),
            # Filter out drogue values that cannot be interpret as floating point values.
            # (e.g. - have more than one decimal point)
            lambda df: df["drogue"].astype(np.str_).str.match(r"(\d+[\.]+){2,}"),
        ],
    )

    drifter_ds_map = dict[int, xr.Dataset]()

    preremove_len = len(preremove_df_chunk)
    postremove_len = len(df_chunk)

    if preremove_len != postremove_len:
        warnings.warn(
            f"Filters removed {preremove_len - postremove_len} rows from chunk"
        )

    if postremove_len == 0:
        return drifter_ds_map

    df_chunk = df_chunk.astype(_INPUT_COLS_DTYPES)
    df_chunk = _apply_transform(
        df_chunk,
        {
            "position_datetime": (
                ["posObsMonth", "posObsDay", "posObsYear"],
                _parse_datetime_with_day_ratio,
            ),
            "sensor_datetime": (
                ["senObsMonth", "senObsDay", "senObsYear"],
                _parse_datetime_with_day_ratio,
            ),
        },
    )

    # Find and process drifters found and documented in the drifter metadata.
    ids_in_data = np.unique(df_chunk[["id"]].values)
    ids_with_md = np.intersect1d(ids_in_data, gdp_metadata_df[["ID"]].values)

    if len(ids_with_md) < len(ids_in_data):
        warnings.warn(
            "Chunk has drifter ids not found in the metadata table. "
            + "Using fill values"
            if use_fill_values
            else "Ignoring data observations"
            + f" for missing metadata ids: {np.setdiff1d(ids_in_data, ids_with_md)}."
        )

    ra = RaggedArray.from_files(
        indices=ids_in_data if use_fill_values else ids_with_md,
        preprocess_func=_preprocess,
        rowsize_func=_rowsize,
        name_coords=_COORDS,
        name_meta=_METADATA_VARS,
        name_data=_DATA_VARS,
        name_dims={"traj": "rows", "obs": "obs"},
        md_df=gdp_metadata_df,
        data_df=df_chunk,
        use_fill_values=use_fill_values,
        tqdm=dict(disable=True),
    )
    ds = ra.to_xarray()

    for id_ in ids_with_md:
        id_f_ds = subset(ds, dict(id=id_), row_dim_name="traj")
        drifter_ds_map[id_] = id_f_ds
    return drifter_ds_map


def _combine_chunked_drifter_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Combines several drifter observations found in separate chunks, ordering them
    by the observations row index.
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
    chunk_size: int,
    tmp_path: str,
    use_fill_values: bool,
    max_chunks: int | None,
) -> list[xr.Dataset]:
    """Parallel process dataset in chunks leveraging multiprocessing."""
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
                names=_INPUT_COLS,
                engine="c",
                compression="gzip",
                chunksize=chunk_size,
            )

            joblist = list[Future]()
            jobmap = dict[Future, pd.DataFrame]()
            for idx, chunk in enumerate(file_chunks):
                if max_chunks is not None and idx >= max_chunks:
                    break
                ajob = ppe.submit(
                    _process_chunk,
                    chunk,
                    start_idx,
                    start_idx + len(chunk),
                    gdp_metadata_df,
                    use_fill_values,
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
                if (exc := ajob.exception()) is not None:
                    chunk = jobmap[ajob]
                    _logger.warn(f"bad chunk detected, exception: {ajob.exception()}")
                    raise exc

                job_drifter_ds_map: dict[int, xr.Dataset] = ajob.result()
                for id_ in job_drifter_ds_map.keys():
                    drifter_ds = job_drifter_ds_map[id_]
                    drifter_chunked_datasets[id_].append(drifter_ds)
                bar.update()

        combine_jobmap = dict[Future, int]()
        for id_ in drifter_chunked_datasets.keys():
            datasets = drifter_chunked_datasets[id_]

            combine_job = ppe.submit(_combine_chunked_drifter_datasets, datasets)
            combine_jobmap[combine_job] = id_

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
        for combine_job in as_completed(combine_jobmap.keys()):
            dataset: xr.Dataset = combine_job.result()
            drifter_datasets.append(dataset)
            bar.update()
        bar.close()
        return drifter_datasets


def to_raggedarray(
    tmp_path: str = _TMP_PATH,
    skip_download: bool = False,
    max: int | None = None,
    chunk_size: int = 100_000,
    use_fill_values: bool = True,
    max_chunks: int | None = None,
) -> xr.Dataset:
    """
    Convert GDP source data into a ragged array format and return it as an xarray Dataset.

    This function processes drifter data from the NOAA GDP (Global Drifter Program) source,
    organizes it into a ragged array format, and returns the resulting dataset. It
    supports downloading, filtering, and parallel processing of the data.

    Args:
        tmp_path (str): Path to the temporary directory for storing downloaded files.
                        Defaults to `_TMP_PATH`.
        skip_download (bool): If True, skips downloading the data and assumes it is
                              already available in `tmp_path`. Defaults to False.
        max (int | None): Maximum number of requests to process for testing purposes.
                          If None, processes all requests. Defaults to None.
        chunk_size (int): Number of observations to process in each chunk. Defaults to 100,000.
        use_fill_values (bool): Whether to use fill values for missing data. Defaults to True.
        max_chunks (int | None): Maximum number of chunks to process. If None, processes all
                                 chunks. Defaults to None.

    Returns:
        xr.Dataset: An xarray Dataset containing the processed GDP drifter data in a
                    ragged array format. The dataset includes both observation and
                    trajectory metadata variables, with appropriate attributes added.

    Raises:
        Any exceptions raised during file operations, data processing, or async tasks
        will propagate to the caller.

    Notes:
        - The function performs parallel processing of drifter data using asyncio.
        - The resulting dataset is sorted by the start date of each drifter.
        - Metadata attributes for variables are added based on predefined mappings.
    """

    os.makedirs(tmp_path, exist_ok=True)

    requests = _get_download_list(tmp_path)

    # Filter down for testing purposes.
    if max:
        requests = [requests[max]]

    # Download necessary data and metadata files.
    if not skip_download:
        download_with_progress(requests)

    gdp_metadata_df = get_gdp_metadata(tmp_path)

    # Run async process to parallelize data processing.
    drifter_datasets = asyncio.run(
        _parallel_get(
            [dst for (_, dst) in requests],
            gdp_metadata_df,
            chunk_size,
            tmp_path,
            use_fill_values,
            max_chunks,
        )
    )

    # Sort the drifters by their start date.
    deploy_date_id_map = {
        ds["id"].data[0]: ds["start_date"].data[0] for ds in drifter_datasets
    }
    deploy_date_sort_key = np.argsort(list(deploy_date_id_map.values()))
    sorted_drifter_datasets = [drifter_datasets[idx] for idx in deploy_date_sort_key]

    # Concatenate drifter data and metadata variables separately.
    obs_ds = xr.concat(
        [ds.drop_dims("traj") for ds in sorted_drifter_datasets],
        dim="obs",
        data_vars=_DATA_VARS,
    )
    traj_ds = xr.concat(
        [ds.drop_dims("obs") for ds in sorted_drifter_datasets],
        dim="traj",
        data_vars=_METADATA_VARS,
    )

    # Merge the separate datasets.
    agg_ds = xr.merge([obs_ds, traj_ds])

    # Add variable metadata.
    for var_name in _DATA_VARS + _METADATA_VARS:
        if var_name in VARS_ATTRS.keys():
            agg_ds[var_name].attrs = VARS_ATTRS[var_name]
    agg_ds.attrs = ATTRS

    return agg_ds

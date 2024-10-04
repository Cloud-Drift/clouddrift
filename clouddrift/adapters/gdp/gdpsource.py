from __future__ import annotations

import datetime
import logging
import os
import tempfile
import warnings
from typing import Callable

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from clouddrift.adapters.gdp import get_gdp_metadata
from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

_DATA_URL = "https://www.aoml.noaa.gov/ftp/pub/phod/pub/pazos/data/shane/sst"
_TMP_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "gdpsource")
_FILENAME_TEMPLATE = "buoydata_{start}_{end}_{suffix}.dat.gz"
_SECONDS_IN_DAY = 86_400

_COORDS = ["id", "position_datetime"]

_DATA_VARS = [
    "latitude",
    "longitude",
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
    "wmo_number": np.nan,
    "program_number": np.nan,
    "buoys_type": "N/A",
    "start_date": np.datetime64("NaT"),
    "start_lat": np.nan,
    "start_lon": np.nan,
    "end_date": np.datetime64("NaT"),
    "end_lat": np.nan,
    "end_lon": np.nan,
    "drogue_off_date": np.datetime64("NaT"),
    "death_code": np.nan,
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
    "posObsMonth": np.float32,
    "posObsDay": np.float64,
    "posObsYear": np.float32,
    "latitude": np.float32,
    "longitude": np.float32,
    "qualityIndex": np.float32,
    "senObsMonth": np.float32,
    "senObsDay": np.float64,
    "senObsYear": np.float32,
    "drogue": np.float32,
    "sst": np.float32,
    "voltage": np.float32,
    "sensor4": np.float32,
    "sensor5": np.float32,
    "sensor6": np.float32,
}

_INPUT_COLS_PREFILTER_DTYPES: dict[str, type[object]] = {
    "posObsMonth": np.str_,
    "posObsYear": np.float64,
    "senObsMonth": np.str_,
    "senObsYear": np.float64,
    "drogue": np.str_,
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
        "comments": "Position datetime derived from the year, month, and day (represented as a ratio of a day) when the geographical coordinates of the drifter were obtained. May differ from sensor_datetime.",
    },
    "sensor_datetime": {
        "comments": "Sensor datetime derived from the year, month, and day (represented as a ratio of a day) when sensor data were recorded.",
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
        "comments": "Definitions vary",
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
        "position_datetime": (
            ["obs"],
            traj_data_df[["position_datetime"]].values.flatten().astype(np.datetime64),
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
            datetime.datetime(year=int(year), month=int(month), day=int(1))
            + datetime.timedelta(days=int(day), seconds=seconds)
        ).timestamp() * 10**9
        values.append(int(dt_ns))
    return np.array(values).astype("datetime64[ns]")


def _process(
    df: dd.DataFrame,
    gdp_metadata_df: pd.DataFrame,
    use_fill_values: bool,
) -> xr.Dataset:
    """Process each dataframe chunk. Return a dictionary mapping each drifter to a unique xarray Dataset."""

    # Transform the initial dataframe filtering out rows with really anomolous values
    # examples include: years in the future, years way in the past before GDP program, etc...
    preremove_df = df.compute()
    df_chunk = _apply_remove(
        preremove_df,
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

    preremove_len = len(preremove_df)
    postremove_len = len(df_chunk)

    if preremove_len != postremove_len:
        warnings.warn(
            f"Filters removed {preremove_len - postremove_len} rows from chunk"
        )

    if postremove_len == 0:
        raise ValueError("All rows removed from dataframe, please review filters")

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

    if use_fill_values:
        selected_ids = ids_in_data
    else:
        selected_ids = ids_with_md

    gdp_start_dates = list()
    for id_ in selected_ids:
        selected_drifter = gdp_metadata_df[gdp_metadata_df["ID"] == id_]

        if len(selected_drifter) == 0:
            gdp_start_dates.append(np.datetime64("NaT"))
        else:
            gdp_start_dates.append(selected_drifter[["Start_date"]].values.flatten()[0])

    start_date_sortkey = np.argsort(gdp_start_dates)
    start_date_sorted_ids = selected_ids[start_date_sortkey]

    ra = RaggedArray.from_files(
        indices=start_date_sorted_ids,
        preprocess_func=_preprocess,
        rowsize_func=_rowsize,
        name_coords=_COORDS,
        name_meta=_METADATA_VARS,
        name_data=_DATA_VARS,
        name_dims={"traj": "rows", "obs": "obs"},
        md_df=gdp_metadata_df,
        data_df=df_chunk,
        use_fill_values=use_fill_values,
        tqdm={"disable": True},
    )
    return ra.to_xarray()


def to_raggedarray(
    tmp_path: str = _TMP_PATH,
    max: int | None = None,
    use_fill_values: bool = True,
) -> xr.Dataset:
    """Get the GDP source dataset."""

    os.makedirs(tmp_path, exist_ok=True)

    requests = _get_download_list(tmp_path)

    # Filter down for testing purposes.
    if max:
        requests = requests[:max]

    # Download necessary data and metadata files.
    download_with_progress(requests)

    gdp_metadata_df = get_gdp_metadata(tmp_path)

    import gzip

    data_files = list()
    for compressed_data_file in tqdm(
        [dst for (_, dst) in requests], desc="Decompressing files", unit="file"
    ):
        decompressed_fp = compressed_data_file[:-3]
        data_files.append(decompressed_fp)
        if not os.path.exists(decompressed_fp):
            with (
                gzip.open(compressed_data_file, "rb") as compr,
                open(decompressed_fp, "wb") as decompr,
            ):
                decompr.write(compr.read())

    wanted_dtypes = dict()
    wanted_dtypes.update(_INPUT_COLS_DTYPES)
    wanted_dtypes.update(_INPUT_COLS_PREFILTER_DTYPES)

    df: dd.DataFrame = dd.read_csv(
        data_files,
        sep=r"\s+",
        header=None,
        names=_INPUT_COLS,
        dtype=wanted_dtypes,
        engine="c",
        blocksize="1GB",
        assume_missing=True,
    )
    ds = _process(df, gdp_metadata_df, use_fill_values)

    # Sort the drifters by their start date.
    # Add variable metadata.
    for var_name in _DATA_VARS + _METADATA_VARS:
        if var_name in VARS_ATTRS.keys():
            ds[var_name].attrs = VARS_ATTRS[var_name]
    ds.attrs = ATTRS

    return ds

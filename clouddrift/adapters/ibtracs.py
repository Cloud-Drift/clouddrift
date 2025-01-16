import os
import tempfile
from typing import Hashable, Literal, TypeAlias

import numpy as np
import xarray as xr

from clouddrift.adapters.utils import download_with_progress
from clouddrift.raggedarray import RaggedArray

_DEFAULT_FILE_PATH = os.path.join(tempfile.gettempdir(), "clouddrift", "ibtracs")

_SOURCE_BASE_URI = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"

_SOURCE_URL_FMT = "{base_uri}/{version}/access/netcdf/IBTrACS.{kind}.{version}.nc"

_Version: TypeAlias = Literal["v04r00"] | Literal["v04r01"]

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


def to_raggedarray(
    version: _Version, kind: _Kind, tmp_path: str = _DEFAULT_FILE_PATH
) -> xr.Dataset:
    """Returns International Best Track Archive for Climate Stewardship (IBTrACS) as a ragged array xarray dataset.

    The upstream data is available at https://www.ncei.noaa.gov/products/international-best-track-archive

    Parameters
    ----------
    version : str, optional
        Specify the dataset version to retrieve. Default to the latest version. Default is "v04r01".
    kind: str, optional
        Specify the dataset kind to retrieve. Specifying the kind can speed up execution time of specific querries
        and operations. Default is "LAST_3_YEARS".
    tmp_path: str, default adapter temp path (default)
        Temporary path where intermediary files are stored. Default is ${osSpecificTempFileLocation}/clouddrift/ibtracs/.

    Returns
    -------
    xarray.Dataset
        IBTRACS dataset as a ragged array.
    """
    ds = _get_original_dataset(version, kind, tmp_path)
    ds = ds.rename_dims({"date_time": "obs"})

    vars = list[Hashable]()
    vars.extend(ds.variables.keys())
    for coord in ds.coords.keys():
        vars.remove(coord)
    dtypes = {v: ds[v].dtype for v in vars}
    dtypes.update({"numobs": np.dtype("int64")})
    ds = ds.astype(dtypes)

    data_vars = list()
    md_vars = list()

    for var_name in ds.variables:
        # time variable shouldn't be considered a data or metadata variable
        if var_name in ["time"]:
            continue

        var: xr.DataArray = ds[var_name]

        if "obs" in var.dims and len(var.dims) >= 2:
            data_vars.append(var_name)
        elif len(var.dims) == 1 and var.dims[0] == "storm":
            md_vars.append(var_name)

    ra = RaggedArray.from_files(
        indices=list(range(0, len(ds["sid"]))),
        name_coords=["id", "time"],
        name_meta=md_vars,
        name_data=data_vars,
        name_dims={"storm": "rows", "obs": "obs", "quadrant": "quadrant"},
        rowsize_func=_rowsize,
        preprocess_func=_preprocess,
        attrs_global=ds.attrs,
        attrs_variables={
            var_name: ds[var_name].attrs for var_name in data_vars + md_vars
        },
        dataset=ds,
        data_vars=data_vars,
        md_vars=md_vars,
    )
    return ra.to_xarray()


def _get_original_dataset(
    version: _Version, kind: _Kind, tmp_path: str = _DEFAULT_FILE_PATH
) -> xr.Dataset:
    os.makedirs(tmp_path, exist_ok=True)
    src_url = _get_source_url(version, kind)

    filename = src_url.split("/")[-1]
    dst_path = os.path.join(tmp_path, filename)
    download_with_progress([(src_url, dst_path)])

    return xr.open_dataset(dst_path, engine="netcdf4")


def _rowsize(idx: int, **kwargs):
    ds: xr.Dataset | None = kwargs.get("dataset")
    if ds is None:
        raise ValueError("kwargs dataset missing")
    storm_ds = ds.isel(storm=idx)
    return storm_ds["numobs"].data


def _preprocess(idx: int, **kwargs):
    ds: xr.Dataset | None = kwargs.get("dataset")
    data_vars: list[str] | None = kwargs.get("data_vars")
    md_vars: list[str] | None = kwargs.get("md_vars")

    if ds is not None and data_vars is not None and md_vars is not None:
        storm_ds = ds.isel(storm=idx)
        numobs = storm_ds["numobs"].data
        vars = dict()

        for var in data_vars + list(storm_ds.coords):
            if var != "time":
                vars.update({var: (storm_ds[var].dims, storm_ds[var].data[:numobs])})

        for var in md_vars:
            vars.update({var: (("storm",), [storm_ds[var].data])})

        return xr.Dataset(
            vars,
            {
                "id": (("storm",), np.array([idx])),
                "time": (("obs",), storm_ds["time"].data[:numobs]),
            },
        )
    else:
        raise ValueError("kwargs dataset, data vars and md_vars missing")


def _kind_map(kind: _Kind):
    return {"LAST_3_YEARS": "last3years", "SINCE_1980": "since1980"}.get(kind, kind)


def _get_source_url(version: _Version, kind: _Kind):
    return _SOURCE_URL_FMT.format(
        base_uri=_SOURCE_BASE_URI, version=version, kind=_kind_map(kind)
    )

"""Module for binning Lagrangian data."""

import datetime
import warnings
from functools import partial, wraps
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_BINS_NUMBER = 10
DEFAULT_COORD_NAME = "coord"
DEFAULT_DATA_NAME = "data"


def binned_statistics(
    coords: np.ndarray | list[np.ndarray],
    data: np.ndarray | list[np.ndarray] | None = None,
    bins: int | list = DEFAULT_BINS_NUMBER,
    bins_range: list | None = None,
    dim_names: list[str] | None = None,
    output_names: list[str] | None = None,
    statistics: str | list | Callable[[np.ndarray], float] = "count",
) -> xr.Dataset:
    """
    Perform N-dimensional binning and compute statistics of values in each bin. The result is returned as an Xarray Dataset.

    Parameters
    ----------
    coords : array-like or list of array-like
        Array(s) of Lagrangian data coordinates to be binned. For 1D, provide a single array.
        For N-dimensions, provide a list of N arrays, each giving coordinates along one dimension.
    data : array-like or list of array-like
        Data values associated with the Lagrangian coordinates in coords.
        Can be a single array or a list of arrays for multiple variables.
        Complex values are supported for the supported statistics except for 'min', 'max', and 'median'.
    bins : int or lists, optional
        Number of bins or bin edges per dimension. It can be:
        - An int: same number of bins for all dimensions,
        - A list of ints or arrays: one per dimension, specifying either bin count or bin edges,
        - None: defaults to 10 bins per dimension.
    bins_range : list of tuples, optional
        Outer bin limits for each dimension.
    statistics : str or list of str, Callable[[np.ndarray], float] or list[Callable[[np.ndarray], float]]
        Statistics to compute for each bin. It can be:
        - a string, supported values: 'count', 'sum', 'mean', 'median', 'std', 'min', 'max', (default: "count"),
        - a custom function as a callable for univariate statistics that take a 1D array of values and return a single value.
          The callable is applied to each variable of data.
        - a tuple of (output_name, callable) for multivariate statistics. 'output_name' is used to identify the resulting variables.
          In this case, the callable will receive the list of arrays provided in `data`. For example, to calculate kinetic energy from data with velocity components `u` and `v`,
          you can pass `data = [u, v]` and  `statistics=("ke", lambda data: np.sqrt(np.mean(data[0] ** 2 + data[1] ** 2)))`.
        - a list containing any combination of the above, e.g., ['mean', np.nanmax, ('ke', lambda data: np.sqrt(np.mean(data[0] ** 2 + data[1] ** 2)))].
    dim_names : list of str, optional
        Names for the dimensions of the output xr.Dataset.
        If None, default names are "coord_0", "coord_1", etc.
    output_names : list of str, optional
        Names for output variables in the xr.Dataset.
        If None, default names are "data_0_{statistic}", "data_1_{statistic}", etc.

    Returns
    -------
    xr.Dataset
        Xarray dataset with binned means and count for each variable.
    """
    # convert coords, data parameters to numpy arrays and validate dimensions
    # D, N = number of dimensions and number of data points
    if not isinstance(coords, np.ndarray) or coords.ndim == 1:
        coords = np.atleast_2d(coords)
    D, N = coords.shape

    # validate coordinates are finite
    for c in coords:
        var = c.copy()
        if var.dtype == "O":
            var = var.astype(type(var[0]))
        if _is_datetime_array(var):
            if pd.isna(var).any():
                raise ValueError("Datetime coordinates must be finite values.")
        else:
            if pd.isna(var).any() or np.isinf(var).any():
                raise ValueError("Coordinates must be finite values.")

    # V, VN = number of variables and number of data points per variable
    if data is None:
        data = np.empty((1, 0))
        V, VN = 1, N  # no data provided
    elif not isinstance(data, np.ndarray) or data.ndim == 1:
        data = np.atleast_2d(data)
        V, VN = data.shape
    else:
        V, VN = data.shape

    # convert datetime coordinates to numeric values
    coords_datetime_index = np.where([_is_datetime_array(c) for c in coords])[0]
    for i in coords_datetime_index:
        coords[i] = _datetime64_to_float(coords[i])
    coords = coords.astype(np.float64)

    # set default bins and bins range
    if isinstance(bins, (list, tuple)):
        if len(bins) != len(coords):
            raise ValueError("`bins` must match the dimensions of the coordinates")
        bins = [b if b is not None else DEFAULT_BINS_NUMBER for b in bins]
    elif isinstance(bins, int):
        bins = [bins if bins is not None else DEFAULT_BINS_NUMBER] * len(coords)

    if bins_range is None:
        bins_range = [(np.nanmin(c), np.nanmax(c)) for c in coords]
    else:
        if isinstance(bins_range, tuple):
            bins_range = [bins_range] * len(coords)
        bins_range = [
            r if r is not None else (np.nanmin(c), np.nanmax(c))
            for r, c in zip(bins_range, coords)
        ]

    # validate statistics parameter
    ordered_statistics = ["count", "sum", "mean", "median", "std", "min", "max"]
    if isinstance(statistics, (str, tuple)) or callable(statistics):
        statistics = [statistics]
    elif not isinstance(statistics, list):
        raise ValueError(
            "`statistics` must be a string, list of strings, Callable, or a list of Callables. "
            f"Supported values: {', '.join(ordered_statistics)}."
        )
    if invalid := [
        stat
        for stat in statistics
        if (stat not in ordered_statistics)
        and not callable(stat)
        and not isinstance(stat, tuple)
    ]:
        raise ValueError(
            f"Unsupported statistic(s): {', '.join(map(str, invalid))}. "
            f"Supported: {ordered_statistics} or a Callable."
        )

    # validate multivariable statistics
    for statistic in statistics:
        if isinstance(statistic, tuple):
            output_name, statistic = statistic
            if not isinstance(output_name, str):
                raise ValueError(
                    f"Invalid output name '{output_name}', must be a string."
                )
            if not callable(statistic):
                raise ValueError(
                    "Multivariable `statistics` function is not Callable, must provide as a tuple(output_name, Callable)."
                )

    # validate and sort statistics for efficiency
    statistics_str = [s for s in statistics if isinstance(s, str)]
    statistics_func = [s for s in statistics if not isinstance(s, str)]
    statistics = (
        sorted(
            set(statistics_str),
            key=lambda x: ordered_statistics.index(x),
        )
        + statistics_func
    )

    if statistics and not data.size:
        warnings.warn(
            f"no `data` provided, `statistics` ({statistics}) will be computed on the coordinates."
        )

    # set default dimension names
    if dim_names is None:
        dim_names = [f"{DEFAULT_COORD_NAME}_{i}" for i in range(len(coords))]
    else:
        dim_names = [
            name if name is not None else f"{DEFAULT_COORD_NAME}_{i}"
            for i, name in enumerate(dim_names)
        ]

    # set default variable names
    if output_names is None:
        output_names = [
            f"{DEFAULT_DATA_NAME}_{i}" if data[0].size else DEFAULT_DATA_NAME
            for i in range(len(data))
        ]
    else:
        output_names = [
            name if name is not None else f"{DEFAULT_DATA_NAME}_{i}"
            for i, name in enumerate(output_names)
        ]

    # ensure inputs are consistent
    if D != len(dim_names):
        raise ValueError("`coords` and `dim_names` must have the same length")
    if V != len(output_names):
        raise ValueError("`data` and `output_names` must have the same length")
    if N != VN:
        raise ValueError("`coords` and `data` must have the same number of data points")

    # edges and bin centers
    if isinstance(bins, int) or isinstance(bins[0], int):
        edges = [np.linspace(r[0], r[1], b + 1) for r, b in zip(bins_range, bins)]
    else:
        edges = [np.asarray(b) for b in bins]
    edges_sz = [len(e) - 1 for e in edges]
    n_bins = int(np.prod(edges_sz))
    bin_centers = [0.5 * (e[:-1] + e[1:]) for e in edges]

    # convert bin centers back to datetime64 for output dataset
    for i in coords_datetime_index:
        bin_centers[i] = _float_to_datetime64(bin_centers[i])

    # digitize coordinates into bin indices
    # modify edges to ensure the last edge is inclusive
    # by adding a small tolerance to the last edge (1s for date coordinates)
    edges_with_tol = [e.copy() for e in edges]
    for i, e in enumerate(edges_with_tol):
        e[-1] += np.finfo(float).eps if i not in coords_datetime_index else 1
    indices = [np.digitize(c, edges_with_tol[j]) - 1 for j, c in enumerate(coords)]
    valid = np.all(
        [(j >= 0) & (j < edges_sz[i]) for i, j in enumerate(indices)], axis=0
    )
    indices = [i[valid] for i in indices]

    # create an iterable of statistics to compute
    statistics_iter = []
    for statistic in statistics:
        if isinstance(statistic, str) or callable(statistic):
            for var, name in zip(data, output_names):
                statistics_iter.append((var, name, statistic))
        elif isinstance(statistic, tuple):
            output_name, statistic = statistic
            statistics_iter.append((data, output_name, statistic))

    ds = xr.Dataset()
    for var, name, statistic in statistics_iter:
        # count the number of points in each bin
        var_finite, indices_finite = _filter_valid_and_finite(var, indices, valid)
        flat_idx = np.ravel_multi_index(indices_finite, edges_sz)

        # convert object arrays to a common type
        if var_finite.dtype == "O":
            var_finite = var_finite.astype(type(var_finite[0]))

        # loop through statistics for the variable
        bin_count, bin_mean, bin_sum = None, None, None

        if statistic == "count":
            binned_stats = _binned_count(flat_idx, n_bins)
            bin_count = binned_stats.copy()
        elif statistic == "sum":
            if _is_datetime_array(var_finite):
                raise ValueError("Datetime data is not supported for 'sum' statistic.")
            binned_stats = _binned_sum(flat_idx, n_bins, values=var_finite)
            bin_sum = binned_stats.copy()
        elif statistic == "mean":
            binned_stats = _binned_mean(
                flat_idx,
                n_bins,
                values=var_finite,
                bin_counts=bin_count,
                bin_sum=bin_sum,
            )
            bin_mean = binned_stats.copy()
        elif statistic == "std":
            binned_stats = _binned_std(
                flat_idx,
                n_bins,
                values=var_finite,
                bin_counts=bin_count,
                bin_mean=bin_mean,
            )
        elif statistic == "min":
            binned_stats = _binned_min(
                flat_idx,
                n_bins,
                values=var_finite,
            )
        elif statistic == "max":
            binned_stats = _binned_max(
                flat_idx,
                n_bins,
                values=var_finite,
            )
        elif statistic == "median":
            if np.iscomplexobj(var_finite):
                raise ValueError(
                    "Complex values are not supported for 'median' statistic."
                )
            binned_stats = _binned_apply_func(
                flat_idx,
                n_bins,
                values=var_finite,
                func=np.median,
            )
        else:
            binned_stats = _binned_apply_func(
                flat_idx,
                n_bins,
                values=var_finite,
                func=statistic,
            )

        # add the binned statistics variable to the Dataset
        variable_name = (
            name
            if var_finite.ndim == 2
            else _get_variable_name(name, statistic, ds.data_vars)
            if callable(statistic)
            else f"{name}_{statistic}"
        )

        ds[variable_name] = xr.DataArray(
            binned_stats.reshape(edges_sz),
            dims=dim_names,
            coords=dict(zip(dim_names, bin_centers)),
        )

    return ds


def _get_variable_name(
    output_name: str,
    func: Callable,
    ds_vars: xr.core.dataset.DataVariables | dict[str, xr.DataArray],
) -> str:
    """
    Get the name of the function or a default name if it is a lambda function.

    Parameters
    ----------
    func : Callable
        Function to get the name of.
    output_name : str
        Name of the output variable to which the function is applied.
    ds_vars : dict[str, xr.DataArray]
        Dictionary of existing variables in the dataset to avoid name collisions.

    Returns
    -------
    str
        Name of the function or a custom function name for lambda function.
    """
    default_name = "stat"
    if isinstance(func, partial):
        function_name = getattr(func.func, "__name__", default_name)
    else:
        function_name = getattr(func, "__name__", default_name)
        if function_name == "<lambda>":
            function_name = default_name

    # avoid name collisions with existing variables
    # by adding a suffix if the name already exists
    base_name = f"{output_name}_{function_name}"
    name = base_name

    i = 1
    while name in ds_vars:
        name = f"{base_name}_{i}"
        i += 1

    return name


def _filter_valid_and_finite(
    var: np.ndarray, indices: list, valid: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Filter valid and finite values from the variable and indices.

    Args:
        var : np.ndarray
            Variable data to filter.
        indices : list
            List of index arrays to filter.
        valid : np.ndarray
            Boolean array indicating valid entries.
        V : int
            Size of the 'data' parameter to determine if the variable is multivariate.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: Filtered variable and indices.
    """
    if var.ndim == 2:
        var_valid = [v[valid] for v in var]
        mask = np.logical_or.reduce([~pd.isna(v) for v in var_valid])
        var_finite = np.array([v[mask] for v in var_valid])
        indices_finite = [i[mask] for i in indices]
    elif var.size:
        var = var[valid]
        mask = ~pd.isna(var)
        var_finite = var[mask]
        indices_finite = [i[mask] for i in indices]
    else:
        var_finite = var.copy()
        indices_finite = indices.copy()

    return var_finite, indices_finite


def _is_datetime_subelement(arr: np.ndarray) -> bool:
    """
    Get the type of the first non-null element in an array.

    Parameters
    ----------
    arr : np.ndarray
        Numpy array to check.

    Returns
    -------
    bool
        True if the first non-null element is a datetime type, False otherwise.
    """
    for item in arr.flat:
        if item is not None:
            return isinstance(item, (datetime.date, np.datetime64))
    return False


def _is_datetime_array(arr: np.ndarray) -> bool:
    """
    Verify if an array contains datetime values.

    Parameters
    ----------
    arr : np.ndarray
        Numpy array to check.

    Returns
    -------
    bool
        True if the array contains datetime64 or timedelta64 values, False otherwise.
    """
    if arr.dtype.kind == "M":  # numpy datetime64
        return True
    # if array is object, check first element
    if arr.dtype == object and arr.size > 0:
        return _is_datetime_subelement(arr)
    return False


def _datetime64_to_float(time_dt: np.ndarray) -> np.ndarray:
    """
    Convert np.datetime64 or array of datetime64 to float time since epoch.

    Parameters:
    ----------
    time_dt : np.datetime64 or array-like
        Datetime64 values to convert.

    Returns:
    -------
    float or np.ndarray of floats
        Seconds since UNIX epoch (1970-01-01T00:00:00).
    """
    reference_date = np.datetime64("1970-01-01T00:00:00")
    return np.array(
        (pd.to_datetime(time_dt) - pd.to_datetime(reference_date))
        / pd.to_timedelta(1, "s")
    )


def _float_to_datetime64(time_float, count=None):
    """
    Convert float seconds since UNIX epoch to np.datetime64.

    Parameters:
    ----------
    time_float : float or array-like
        Seconds since epoch (1970-01-01T00:00:00).

    Returns:
    -------
    np.datetime64 or np.ndarray of np.datetime64
        Converted datetime64 values.
    """
    reference_date = np.datetime64("1970-01-01T00:00:00")
    date = reference_date + time_float.astype("timedelta64[s]")

    return date


def handle_datetime_conversion(func: Callable) -> Callable:
    """
    A decorator to handle datetime64/timedelta64 conversion for
    statistics functions. For datetime `values`, it converts the time to float
    seconds since epoch before calling the function, and converts the result back
    to datetime64 after the function call.

    Assumes that the function accepts `values` as keyword arguments.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> np.ndarray:
        values = kwargs.get("values")

        datetime_conversion = False
        if values is not None:
            if datetime_conversion := _is_datetime_array(values):
                kwargs["values"] = _datetime64_to_float(values)

        # call back the original function
        result = func(*args, **kwargs)

        # Convert the result to datetime if necessary
        if datetime_conversion:
            if func.__name__ == "_binned_std":
                return result.astype("timedelta64[s]")
            return _float_to_datetime64(result)

        return result

    return wrapper


def _binned_count(flat_idx: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute the count of values in each bin.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices, same shape as values.
    n_bins: int
        number of bins

    Returns
    -------
    result : array-like
        1D array of length n_bins with the count per bin
    """
    return np.bincount(flat_idx, minlength=n_bins)


@handle_datetime_conversion
def _binned_sum(flat_idx: np.ndarray, n_bins: int, values: np.ndarray) -> np.ndarray:
    """
    Compute the sum of values per bin.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices, same shape as values.
    n_bins: int
        number of bins
    values : array-like
        1D array of data values

    Returns
    -------
    result : array-like
        1D array of length n_bins with the sum per bin
    """
    if np.iscomplexobj(values):
        real = np.bincount(flat_idx, weights=values.real, minlength=n_bins)
        imag = np.bincount(flat_idx, weights=values.imag, minlength=n_bins)
        return real + 1j * imag
    else:
        return np.bincount(flat_idx, weights=values, minlength=n_bins)


@handle_datetime_conversion
def _binned_mean(
    flat_idx: np.ndarray,
    n_bins: int,
    values: np.ndarray,
    bin_counts: np.ndarray | None = None,
    bin_sum: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the mean of values per bin.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices, same shape as values.
    n_bins: int
        number of bins
    values : array-like
        1D array of data values
    bin_counts : array-like, optional
        Precomputed counts per bin. If None, it will be computed using `_binned_count`.
    bin_sum : array-like, optional
        Precomputed sum per bin. If None, it will be computed using `_binned_sum`.

    Returns
    -------
    result : array-like
        1D array of length n_bins with the mean per bin
    """
    if bin_counts is None:
        bin_counts = _binned_count(flat_idx, n_bins)

    if bin_sum is None:
        bin_sum = _binned_sum(flat_idx, n_bins, values)

    return np.divide(
        bin_sum,
        bin_counts,
        out=np.full_like(bin_sum, np.nan, dtype=bin_sum.dtype),
        where=bin_counts > 0,
    )


@handle_datetime_conversion
def _binned_std(
    flat_idx: np.ndarray,
    n_bins: int,
    values: np.ndarray,
    bin_counts: np.ndarray | None = None,
    bin_mean: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the standard deviation of values per bin.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices, same shape as values.
    n_bins: int
        number of bins
    values : array-like
        1D array of data values
    bin_counts : array-like, optional
        Precomputed counts per bin. If None, it will be computed using `_binned_count`.
    bin_mean : array-like, optional
        Precomputed mean per bin. If None, it will be computed using `_binned_mean`.

    Returns
    -------
    result : array-like
        1D array of length n_bins with the standard deviation per bin
    """
    if bin_counts is None:
        bin_counts = _binned_count(flat_idx, n_bins)

    if bin_mean is None:
        bin_mean = _binned_mean(flat_idx, n_bins, values, bin_counts)

    if np.iscomplexobj(values):
        # Use modulus for variance
        abs_values = np.abs(values)
        bin_sumsq = np.bincount(flat_idx, weights=abs_values**2, minlength=n_bins)
        bin_mean_sq = np.divide(
            bin_sumsq,
            bin_counts,
            out=np.full(n_bins, np.nan, dtype=bin_sumsq.dtype),
            where=bin_counts > 0,
        )
        abs_bin_mean = np.abs(bin_mean)
        variance = np.maximum(bin_mean_sq - abs_bin_mean**2, 0)
    else:
        bin_sumsq = np.bincount(flat_idx, weights=values**2, minlength=n_bins)
        bin_mean_sq = np.divide(
            bin_sumsq,
            bin_counts,
            out=np.full(n_bins, np.nan, dtype=bin_sumsq.dtype),
            where=bin_counts > 0,
        )
        variance = np.maximum(bin_mean_sq - bin_mean**2, 0)

    return np.sqrt(variance)


@handle_datetime_conversion
def _binned_min(flat_idx: np.ndarray, n_bins: int, values: np.ndarray) -> np.ndarray:
    """
    Compute the minimum of values per bin.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices, same shape as values.
    n_bins: int
        number of bins
    values : array-like
        1D array of data values

    Returns
    -------
    result : array-like
        1D array of length n_bins with the minimum per bin
    """
    if np.iscomplexobj(values):
        raise ValueError("Complex values are not supported for 'min' statistic.")

    output = np.full(n_bins, np.inf)
    np.minimum.at(output, flat_idx, values)
    output[output == np.inf] = np.nan
    return output


@handle_datetime_conversion
def _binned_max(flat_idx: np.ndarray, n_bins: int, values: np.ndarray) -> np.ndarray:
    """
    Compute the maximum of values per bin.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices, same shape as values.
    n_bins: int
        number of bins
    values : array-like
        1D array of data values

    Returns
    -------
    result : array-like
        1D array of length n_bins with the maximum per bin
    """
    if np.iscomplexobj(values):
        raise ValueError("Complex values are not supported for 'max' statistic.")

    output = np.full(n_bins, -np.inf)
    np.maximum.at(output, flat_idx, values)
    output[output == -np.inf] = np.nan
    return output


@handle_datetime_conversion
def _binned_apply_func(
    flat_idx: np.ndarray,
    n_bins: int,
    values: np.ndarray,
    func: Callable[[np.ndarray | list[np.ndarray]], float] = np.mean,
) -> np.ndarray:
    """
    Generic wrapper to apply any functions (e.g., percentile) to binned data.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices.
    n_bins : int
        Number of bins.
    values : array-like or list of array-like
        1D array (univariate) or list of 1D arrays (multivariate) of data values.
    func : Callable[[list[np.ndarray]], float]
        Function to apply to each bin. If multivariate, will receive a list of arrays.

    Returns
    -------
    result : np.ndarray
        1D array of length n_bins with results from func per bin.
    """
    sort_indices = np.argsort(flat_idx)
    sorted_flat_idx = flat_idx[sort_indices]

    # single or all variables can be passed as input values
    if is_multivariate := values.ndim == 2:
        sorted_values = [v[sort_indices] for v in values]
    else:
        sorted_values = [values[sort_indices]]

    unique_bins, bin_starts = np.unique(sorted_flat_idx, return_index=True)
    bin_ends = np.append(bin_starts[1:], len(sorted_flat_idx))

    result = np.full(n_bins, np.nan)
    for i, bin_idx in enumerate(unique_bins):
        if is_multivariate:
            bin_values = [v[bin_starts[i] : bin_ends[i]] for v in sorted_values]
        else:
            bin_values = sorted_values[0][bin_starts[i] : bin_ends[i]]
        result[bin_idx] = func(bin_values)

    return result

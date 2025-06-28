"""Module for binning Lagrangian data."""

import functools
from typing import Callable

import numpy as np
import xarray as xr

DEFAULT_BINS_NUMBER = 10


def _get_function_name(func: Callable[[np.ndarray], float]) -> str:
    """
    Get the name of the function or a default name if it is a lambda function.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Function to get the name of.

    Returns
    -------
    str
        Name of the function or a custom function name for lambda function.
    """
    function_name = "unknown_callable"
    if isinstance(func, functools.partial):
        function_name = getattr(func.func, "__name__", "partial_wrapper_func")
        if function_name == "<lambda>":
            function_name = "partial_lambda_wrapper"
    elif hasattr(func, "__name__"):
        function_name = func.__name__
        if function_name == "<lambda>":
            function_name = "anonymous_lambda_func"
    return function_name


def _binned_count(flat_idx, n_bins):
    """Compute the count of values in each bin.

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


def _binned_sum(flat_idx, n_bins, values):
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
    return np.bincount(flat_idx, weights=values, minlength=n_bins)


def _binned_mean(flat_idx, n_bins, values, bin_counts=None, bin_sum=None):
    """
    Compute a reduction (mean, std, min, max, etc.) of values per bin.

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
        out=np.full_like(bin_sum, 0.0),
        where=bin_counts > 0,
    )


def _binned_std(flat_idx, n_bins, values, bin_counts=None, bin_mean=None):
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

    bin_sumsq = np.bincount(flat_idx, weights=values**2, minlength=n_bins)
    bin_mean_sq = np.divide(
        bin_sumsq, bin_counts, out=np.full(n_bins, 0.0), where=bin_counts > 0
    )
    variance = np.maximum(bin_mean_sq - bin_mean**2, 0)

    return np.sqrt(variance)


def _binned_min(flat_idx, n_bins, values, fill_value=np.nan):
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
    fill_value : float, optional
        Value to fill in bins with no data. Default is np.nan.

    Returns
    -------
    result : array-like
        1D array of length n_bins with the minimum per bin
    """
    output = np.full(n_bins, np.inf)
    np.minimum.at(output, flat_idx, values)
    output[output == np.inf] = fill_value
    return output


def _binned_max(flat_idx, n_bins, values, fill_value=np.nan):
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
    fill_value : float, optional
        Value to fill in bins with no data. Default is np.nan.

    Returns
    -------
    result : array-like
        1D array of length n_bins with the maximum per bin
    """
    output = np.full(n_bins, -np.inf)
    np.maximum.at(output, flat_idx, values)
    output[output == np.inf] = fill_value
    return output


def _binned_apply_func(
    flat_idx: np.ndarray,
    n_bins: int,
    values: np.ndarray,
    func: Callable[[np.ndarray], float] = np.mean,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Generic wrapper to apply any functions (like median, percentile) to binned data.

    Parameters
    ----------
    flat_idx : array-like
        1D array of bin indices, same shape as values.
    n_bins: int
        number of bins
    values : array-like
        1D array of data values
    func : Callable[[np.ndarray], float], optional
        Function to apply to the values in each bin. Default is np.mean.
    fill_value : float, optional
        Value to fill in bins with no data. Default is np.nan.

    Returns
    -------
    result  : array-like
        1D array of length n_bins with the applied function per bin
    """
    # sort the flat_idx and values
    sort_indices = np.argsort(flat_idx)
    sorted_flat_idx = flat_idx[sort_indices]
    sorted_values = values[sort_indices]

    # find start and end indices for each bin
    unique_bins, bin_starts = np.unique(sorted_flat_idx, return_index=True)
    bin_ends = np.append(bin_starts[1:], len(sorted_flat_idx))

    result = np.full(n_bins, fill_value)
    for i, bin_idx in enumerate(unique_bins):
        bin_values = sorted_values[bin_starts[i] : bin_ends[i]]
        result[bin_idx] = func(bin_values)

    return result


def binned_statistics(
    coords: np.ndarray | list[np.ndarray],
    data: np.ndarray | list[np.ndarray] | None = None,
    bins: int | list = DEFAULT_BINS_NUMBER,
    bins_range: list | None = None,
    dim_names: list[str] | None = None,
    output_names: list[str] | None = None,
    statistics: str | list = "mean",
    functions: Callable[[np.ndarray], float]
    | list[Callable[[np.ndarray], float]]
    | None = None,
    zeros_to_nan: bool = False,
) -> xr.Dataset:
    """
    Perform N-dimensional binning and compute mean of values in each bin. The result is returned as an Xarray Dataset.

    Parameters
    ----------
    coords : array-like or list of array-like
        Array(s) of Lagrangian data coordinates to be binned. For 1D, provide a single array.
        For N-dimensions, provide a list of N arrays, each giving coordinates along one dimension.
    data : array-like or list of array-like
        Data values associated with the Lagrangian coordinates in coords.
        Can be a single array or a list of arrays for multiple variables.
    bins : int or lists, optional
        Number of bins or bin edges per dimension. It can be:
        - An int: same number of bins for all dimensions (default: 10),
        - A list of ints or arrays: one per dimension, specifying either bin count or bin edges,
        - None: defaults to 10 bins per dimension.
    bins_range : list of tuples, optional
        Outer bin limits for each dimension.
    statistics : str or list of str, optional
        Statistics to compute for each bin (default: "mean"). Can be a string or a list of
        variables to compute. Supported values are 'count', 'sum', 'mean', 'median', 'std', 'min', 'max'.
    functions : Callable[[np.ndarray], float] or list of Callable[[np.ndarray], float], optional
        Custom functions to apply to the binned data. Each function should take a 1D array
        of values and return a single value.
    dim_names : list of str, optional
        Names for the dimensions of the output xr.Dataset.
        If None, default names are "dim_0_bin", "dim_1_bin", etc.
    output_names : list of str, optional
        Names for output variables in the xr.Dataset.
        If None, default names are "binned_mean_0", "binned_mean_1", etc.
    zeros_to_nan : bool, optional
        If True, replace zeros in the output(s) with NaN. Default is False.

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

    # V, VN = number of variables and number of data points per variable
    if data is None:
        data = np.empty((1, 0))
        V, VN = 1, N  # no data provided
    elif not isinstance(data, np.ndarray) or data.ndim == 1:
        data = np.atleast_2d(data)
        V, VN = data.shape
    else:
        V, VN = data.shape

    # set default bins and bins range
    if isinstance(bins, (list, tuple)):
        if len(bins) != len(coords):
            raise ValueError("'bins' must match the dimensions of the coordinates")
        bins = [b if b is not None else DEFAULT_BINS_NUMBER for b in bins]
    elif isinstance(bins, int):
        bins = [bins if bins is not None else DEFAULT_BINS_NUMBER] * len(coords)

    if bins_range is None:
        bins_range = [
            (np.nanmin(c), np.nanmax(c) + np.finfo(np.float64).eps) for c in coords
        ]
    else:
        if isinstance(bins_range, tuple):
            bins_range = [bins_range] * len(coords)
        bins_range = [
            r if r is not None else (np.nanmin(c), np.nanmax(c))
            for r, c in zip(bins_range, coords)
        ]

    # create a list if statistics is a string
    ordered_statistics = ["count", "sum", "mean", "std", "median", "min", "max"]
    if isinstance(statistics, str):
        statistics = [statistics]
    elif not isinstance(statistics, (list, tuple)):
        raise ValueError(
            "'statistics' must be a string or a list of strings. "
            f"Supported values: {', '.join(ordered_statistics)}."
        )

    if invalid := [stat for stat in statistics if stat not in ordered_statistics]:
        raise ValueError(
            f"Unsupported statistic(s): {', '.join(invalid)}. "
            f"Supported values: {', '.join(ordered_statistics)}."
        )
    statistics = sorted(set(statistics), key=lambda x: ordered_statistics.index(x))

    # set default dimension names
    if dim_names is None:
        dim_names = [f"dim_{i}_bin" for i in range(len(coords))]
    else:
        dim_names = [
            name if name is not None else f"dim_{i}_bin"
            for i, name in enumerate(dim_names)
        ]

    # set default variable names
    if output_names is None:
        output_names = [
            f"binned_{i}" if data[0].size else "binned_count" for i in range(len(data))
        ]
    else:
        output_names = [
            name if name is not None else f"binned_{i}"
            for i, name in enumerate(output_names)
        ]

    # ensure inputs are consistent
    if D != len(dim_names):
        raise ValueError("'coords' and 'dim_names' must have the same length")
    if V != len(output_names):
        raise ValueError("'data' and 'output_names' must have the same length")
    if N != VN:
        raise ValueError("'coords' and 'data' must have the same number of data points")

    # edges and bin centers
    edges = [np.linspace(r[0], r[1], b + 1) for r, b in zip(bins_range, bins)]
    edges_sz = [len(e) - 1 for e in edges]
    bin_centers = [0.5 * (e[:-1] + e[1:]) for e in edges]

    # digitize coordinates into bin indices
    indices = [np.digitize(c, edges[j]) - 1 for j, c in enumerate(coords)]
    valid = np.all(
        [(j >= 0) & (j < edges_sz[i]) for i, j in enumerate(indices)], axis=0
    )
    indices = [i[valid] for i in indices]

    ds = xr.Dataset()
    for var, name in zip(data, output_names):
        if var.size:
            var = var[valid]
            mask = np.isfinite(var)
            var_finite = var[mask]
            indices_finite = [i[mask] for i in indices]
        else:
            indices_finite = indices.copy()

        # count the number of points in each bin
        flat_idx = np.ravel_multi_index(indices_finite, edges_sz)
        bin_counts = _binned_count(flat_idx, np.prod(edges_sz))
        bin_mean, bin_sum = None, None

        # add bin count to the dataset
        if zeros_to_nan and np.any(bin_counts == 0):
            bin_counts = np.where(bin_counts == 0, np.nan, bin_counts)

        count_var_name = f"{name}_count" if var.size else f"{name}"
        ds[count_var_name] = xr.DataArray(
            bin_counts.reshape(edges_sz),
            dims=dim_names,
            coords=dict(zip(dim_names, bin_centers)),
        )

        if not var.size:
            return ds

        # loop through statistics for the variable
        for statistic in statistics:
            print(f"Computing {statistic} for variable '{name}'...")
            if statistic == "sum":
                binned_stats = _binned_sum(flat_idx, np.prod(edges_sz), var_finite)
                bin_sum = binned_stats
            elif statistic == "mean":
                binned_stats = _binned_mean(
                    flat_idx,
                    np.prod(edges_sz),
                    var_finite,
                    bin_sum=bin_sum,
                    bin_counts=bin_counts,
                )
                bin_mean = binned_stats
            elif statistic == "std":
                binned_stats = _binned_std(
                    flat_idx,
                    np.prod(edges_sz),
                    var_finite,
                    bin_mean=bin_mean,
                    bin_counts=bin_counts,
                )
            elif statistic == "min":
                binned_stats = _binned_min(
                    flat_idx,
                    np.prod(edges_sz),
                    var_finite,
                    fill_value=np.nan,
                )
            elif statistic == "max":
                binned_stats = _binned_max(
                    flat_idx,
                    np.prod(edges_sz),
                    var_finite,
                    fill_value=np.nan,
                )

            if statistic != "count":
                if zeros_to_nan and np.any(binned_stats == 0):
                    binned_stats = np.where(binned_stats == 0, np.nan, binned_stats)

                # and variable to the Dataset
                ds[f"{name}_{statistic}"] = xr.DataArray(
                    binned_stats.reshape(edges_sz),
                    dims=dim_names,
                    coords=dict(zip(dim_names, bin_centers)),
                )

        # loop through custom functions for the variable
        for function in functions:
            print(
                f"Applying custom function '{_get_function_name(function)}' for variable '{name}'..."
            )
            binned_stats = _binned_apply_func(
                flat_idx,
                np.prod(edges_sz),
                var_finite,
                func=function,
                fill_value=np.nan,
            )
            if zeros_to_nan and np.any(binned_stats == 0):
                binned_stats = np.where(binned_stats == 0, np.nan, binned_stats)

            # add variable to the Dataset
            function_name = _get_function_name(function)
            ds[f"{name}_{function_name}"] = xr.DataArray(
                binned_stats.reshape(edges_sz),
                dims=dim_names,
                coords=dict(zip(dim_names, bin_centers)),
            )

    return ds

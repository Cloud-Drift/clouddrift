"""Module for binning Lagrangian data."""

import numpy as np
import xarray as xr

DEFAULT_BINS_NUMBER = 10


def binned_statistics(
    coords: np.ndarray | list[np.ndarray],
    data: np.ndarray | list[np.ndarray] | None = None,
    bins: int | list = DEFAULT_BINS_NUMBER,
    bins_range: list | None = None,
    dim_names: list[str] | None = None,
    output_names: list[str] | None = None,
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
            raise ValueError("bins must match the dimensions of the coordinates")
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
            f"binned_mean_{i}" if data[0].size else "binned_count"
            for i in range(len(data))
        ]
    else:
        output_names = [
            name if name is not None else f"binned_mean_{i}"
            for i, name in enumerate(output_names)
        ]

    # ensure inputs are consistent
    if D != len(dim_names):
        raise ValueError("coords and dim_names must have the same length")
    if V != len(output_names):
        raise ValueError("data and output_names must have the same length")
    if N != VN:
        raise ValueError("coords and data must have the same number of data points")

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

            # weighted sum and counts in each bin
            flat_idx = np.ravel_multi_index(indices_finite, edges_sz)
            bin_counts = np.bincount(flat_idx, minlength=np.prod(edges_sz))
            weighted_sum = np.bincount(
                flat_idx, weights=var_finite, minlength=np.prod(edges_sz)
            )

            mean = np.divide(
                weighted_sum,
                bin_counts,
                out=np.full_like(weighted_sum, np.nan if zeros_to_nan else 0.0),
                where=bin_counts > 0,
            )
        else:
            # if no data is provided histogram by counting the coords
            flat_idx = np.ravel_multi_index(indices, edges_sz)
            bin_counts = np.bincount(flat_idx, minlength=np.prod(edges_sz))
            if zeros_to_nan and np.any(bin_counts == 0):
                bin_counts = np.where(bin_counts == 0, np.nan, bin_counts)

        # add variable(s) to dataset
        # add bin count
        count_var_name = f"{name}_count" if var.size else f"{name}"
        ds[count_var_name] = xr.DataArray(
            bin_counts.reshape(edges_sz),
            dims=dim_names,
            coords=dict(zip(dim_names, bin_centers)),
        )

        # and statistic when data is provided
        if var.size:
            ds[name] = xr.DataArray(
                mean.reshape(edges_sz),
                dims=dim_names,
                coords=dict(zip(dim_names, bin_centers)),
            )

    return ds

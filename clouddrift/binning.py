"""Module for binning Lagrangian data."""

import numpy as np
import xarray as xr

DEFAULT_BINS_NUMBER = 10


def histogram(
    coords: np.ndarray | list[np.ndarray],
    data: np.ndarray | list[np.ndarray] = [np.empty(0)],
    bins: int | list = DEFAULT_BINS_NUMBER,
    bins_range: list | None = None,
    dim_names: list[str] | None = None,
    output_names: list[str] | None = None,
    zeros_to_nan: bool = False,
) -> xr.Dataset:
    """
    Compute N-dimensional histogram binning and calculate variable means in each bin.

    Parameters
    ----------
    coords : array-like or list of array-like
        Arrays of the coordinates of the Lagrangian data to be binned. For 1D data, a single array can be provided.
        For multiple dimensions, the first array represents each coordinates along the first dimension,
        the second array represents each coordinates along the second dimension, and so on.
    data : array-like or list of array-like
        Data associated at the Lagrangian coordinates described in coords. Multiple variables can be provided as a list.
    bins : int or lists, optional
        Number of bins per dimension (int) or bin edges per dimension (list). Default is 10.
        If an integer is provided, it will be used for all dimensions.
        If a list is provided, it should match the number of dimensions in coords.
        Each element can be an integer or an array of bin edges.
        If None, defaults to 10 bins per dimension.
    bins_range : list of tuples, optional
        Outer bin limits for each dimension
    dim_names : list of str, optional
        Names for the dimensions of the output xr.Dataset
        If None, default names are "dim_0_bin", "dim_1_bin", etc.
    output_names : list of str, optional
        Names for output variables in the xr.Dataset.
        If None, default names are "binned_mean_0", "binned_mean_1", etc.
    zeros_to_nan : bool, optional
        If True, replace zeros in the output(s) with NaN.

    Returns
    -------
    xr.Dataset
        Dataset with binned means for each variable
    """
    # convert inputs to numpy arrays
    if not isinstance(coords[0], (np.ndarray, list)):
        coords = [coords]
    coords = np.asarray([np.asarray(c) for c in coords])
    if not isinstance(data[0], (np.ndarray, list)):
        data = [data]
    data = [np.asarray(v) for v in data]

    # set default bins and bins range
    if isinstance(bins, (list, tuple)):
        if len(bins) != len(coords):
            raise ValueError("bins must match the number of coordinate dimensions")
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
        output_names = [f"binned_mean_{i}" for i in range(len(data))]
    else:
        output_names = [
            name if name is not None else f"binned_mean_{i}"
            for i, name in enumerate(output_names)
        ]

    # ensure inputs are consistent
    if len(coords) != len(dim_names):
        raise ValueError("coords_list and dim_names must have the same length")
    if len(data) != len(output_names):
        raise ValueError("variables_list and new_names must have the same length")

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
            flat_idx = np.ravel_multi_index(tuple(indices_finite), tuple(edges_sz))
            weighted_sum = np.bincount(
                flat_idx, weights=var_finite, minlength=np.prod(edges_sz)
            )
            bin_counts = np.bincount(flat_idx, minlength=np.prod(edges_sz))

            mean = np.divide(
                weighted_sum,
                bin_counts,
                out=np.full_like(weighted_sum, np.nan if zeros_to_nan else 0.0),
                where=bin_counts > 0,
            )
        else:
            # if no variable is provided histogram by counting the coords
            flat_idx = np.ravel_multi_index(tuple(indices), tuple(edges_sz))
            mean = np.bincount(flat_idx, minlength=np.prod(edges_sz))
            if zeros_to_nan and np.any(mean == 0):
                mean = np.where(mean == 0, np.nan, mean)

        # add variable to dataset
        ds[name] = xr.DataArray(
            mean.reshape(tuple(edges_sz)),
            dims=dim_names,
            coords=dict(zip(dim_names, bin_centers)),
        )

    return ds

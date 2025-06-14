"""Module for conversion between Lagrangian and Eulerian representations of data."""

import numpy as np
import xarray as xr


def binned_nd_average(
    coords_list: list[np.array],
    vars_list: list[np.array] = None,
    bins: int | list = 10,
    bins_range: list | None = None,
    dim_names: list[str] | None = None,
    new_names: list[str] | None = None,
):
    """
    Compute N-dimensional histogram binning and calculate variable means in each bin.

    Parameters
    ----------
    coords_list : list of array-like
        Coordinate arrays for each dimension of the binning space
    vars_list : list of array-like
        Variables to average in each bin
    bins : int or list
        Number of bins per dimension (int) or bin edges per dimension (list)
    bins_range : list of tuples, optional
        Outer bin edges for each dimension
    dim_names : list of str, optional
        Names for the dimensions of the output DataArrays
        If None, default names are "dim_0_bin", "dim_1_bin", etc.
    new_names : list of str, optional
        Names for output DataArrays
        If None, default names are "binned_mean_0", "binned_mean_1", etc.

    Returns
    -------
    xr.Dataset
        Dataset with binned means for each variable
    """
    # Convert inputs to numpy arrays
    coords = [np.asarray(c) for c in coords_list]
    vars_list = [np.asarray(v) for v in vars_list] if vars_list else [None]

    # Set default dimension names
    if dim_names is None:
        dim_names = [f"dim_{i}_bin" for i in range(len(coords_list))]
    else:
        dim_names = [
            name if name is not None else f"dim_{i}_bin"
            for i, name in enumerate(dim_names)
        ]

    # Set default variable names
    if new_names is None:
        new_names = [f"binned_mean_{i}" for i in range(len(vars_list))]
    else:
        new_names = [
            name if name is not None else f"binned_mean_{i}"
            for i, name in enumerate(new_names)
        ]

    # Set default bin ranges
    if bins_range is None:
        bins_range = [(np.nanmin(c), np.nanmax(c)) for c in coords_list]
    else:
        bins_range = [
            r if r is not None else (np.nanmin(c), np.nanmax(c))
            for r, c in zip(bins_range, coords_list)
        ]

    # Ensure inputs are consistent
    if len(coords_list) != len(dim_names):
        raise ValueError("coords_list and dim_names must have the same length")
    if len(vars_list) != len(new_names):
        raise ValueError("vars_list and new_names must have the same length")
    if len(bins_range) != len(coords_list):
        raise ValueError("bins_range must match the number of coordinate dimensions")

    # Create output dataset
    ds = xr.Dataset()

    stacked_coords = np.column_stack(coords)
    for var, name in zip(vars_list, new_names):
        # Calculate weighted sums and counts
        if var is not None:
            mask = ~np.isnan(var)
            weighted_sum, edges = np.histogramdd(
                stacked_coords[mask], bins=bins, range=bins_range, weights=var[mask]
            )

            bin_counts, _ = np.histogramdd(
                stacked_coords[mask], bins=bins, range=bins_range
            )

            # Compute means with NaN for empty bins
            mean = np.divide(
                weighted_sum,
                bin_counts,
                out=np.full_like(weighted_sum, np.nan),
                where=bin_counts > 0,
            )
        else:
            # If no variables provided, regular histogram with coords
            mean, edges = np.histogramdd(stacked_coords, bins=bins, range=bins_range)
            mean[mean == 0] = np.nan  # Set empty bins to NaN

        bin_centers = [0.5 * (e[:-1] + e[1:]) for e in edges]

        # Build coordinates dictionary
        coords_dict = dict(zip(dim_names, bin_centers))

        # Add to dataset
        ds[name] = xr.DataArray(mean, dims=dim_names, coords=coords_dict)

    return ds

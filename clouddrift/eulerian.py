"""Module for conversion between Lagrangian and Eulerian representations of data."""

import numpy as np
import pandas as pd
import xarray as xr


def temporal_grouping(ds=None, x=None, y=None, t=None, time_dim="t", freq="MS"):
    """_summary_

    Args:
        ds (_type_, optional): _description_. Defaults to None.
        x (_type_, optional): _description_. Defaults to None.
        y (_type_, optional): _description_. Defaults to None.
        t (_type_, optional): _description_. Defaults to None.
        time_dim (str, optional): _description_. Defaults to "t".
        freq (str, optional): _description_. Defaults to "MS".
            MS (monthly), AS/YS (yearly), W (weekly), D (daily).

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # create dataset if arrays are provided
    if ds is None:
        if any(v is None for v in (x, y, t)):
            raise ValueError("Either provide Dataset or x/y/t arrays")

        ds = xr.Dataset(
            data_vars={"x": (time_dim, x), "y": (time_dim, y)},
            coords={time_dim: pd.to_datetime(t)},
        )

    # Create temporal bins using pandas timestamp operations
    time_values = ds[time_dim].values
    min_time = pd.Timestamp(time_values.min())
    max_time = pd.Timestamp(time_values.max())

    # Generate month starts and calculate end of last bin
    month_starts = pd.date_range(min_time, max_time, freq=freq)
    if month_starts.empty:
        month_starts = pd.date_range(min_time, periods=1, freq=freq)

    # Properly handle the last bin edge
    last_bin_end = (
        month_starts[-1] + pd.offsets.MonthEnd(1)
    ).to_datetime64() + np.timedelta64(1, "ns")
    bins = np.concatenate([month_starts.values, [last_bin_end]])

    # Convert frequency to period-compatible format
    period_freq_map = {
        "MS": "M",  # Month start → month period
        "YS": "Y",  # Year start → year period
        "AS": "Y",  # Alias for year start
        "W": "W",  # Weekly
        "D": "D",  # Daily
    }
    period_freq = period_freq_map.get(freq, freq)

    # Format labels based on frequency
    periods = pd.period_range(min_time, max_time, freq=period_freq)
    if freq in ["MS", "M"]:
        labels = periods.strftime("%Y-%m")  # 2024-01
    elif freq in ["YS", "AS", "Y"]:
        labels = periods.strftime("%Y")  # 2024
    elif freq.startswith("W"):
        labels = periods.strftime("%Y-W%U")  # 2024-W23
    elif freq == "D":
        labels = periods.strftime("%Y-%m-%d")  # 2024-01-05
    else:
        labels = periods.strftime("%Y-%m-%d")  # Default

    # Assign labels and group
    ds["temporal_group"] = xr.DataArray(
        pd.cut(time_values, bins=bins, labels=labels),
        dims=time_dim,
        coords={time_dim: ds[time_dim]},
    )

    return ds.groupby("temporal_group")


def binned_2d_average(
    x: np.array,
    y: np.array,
    vars_list: list,
    bins: int | list = 10,
    range: list | None = None,
    new_names: list | None = None,
):
    """
    Compute 2D histogram binning of (x, y) and calculate the mean of each variable in vars_list in each bin.
    Constructs an xarray Dataset inside the function.

    Parameters
    ----------
    x, y : array-like
        Coordinates for spatial binning.
    vars_list : list of array-like
        List of variable arrays to average in each bin.
    bins: int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (10, by default).
        If bins is a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge,
        allowing for non-uniform bin widths.
    range: A sequence of length D, each an optional (lower, upper) tuple giving the outer bin edges to be used
        if the edges are not given explicitly in bins. An entry of None in the sequence results in the minimum
        and maximum values being used for the corresponding dimension.
    new_names : list of str, optional
        Names for the new binned mean DataArrays. If None, default names are assigned.

    Returns
    -------
    xr.Dataset
        Dataset containing binned means for each variable with dimensions (x_bin, y_bin).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if new_names is None:
        new_names = [f"binned_mean_{i}" for i in range(len(vars_list))]

    # fix the range if not provided
    # so it is consistent for all variables
    if range is None:
        range = [
            (min(x), max(x)),
            (min(y), max(y)),
        ]

    # Compute the bin edges for output coordinates
    ds = xr.Dataset()
    for i, var in enumerate(vars_list):
        var = np.asarray(var)
        mask = ~np.isnan(var)

        # weighted sum and counts in each bin
        weighted_sum, xedges, yedges = np.histogram2d(
            x[mask], y[mask], bins=bins, range=range, weights=var[mask]
        )
        bin_counts, _, _ = np.histogram2d(
            x[mask],
            y[mask],
            bins=bins,
            range=range,
        )

        mean = np.divide(
            weighted_sum,
            bin_counts,
            out=np.full_like(weighted_sum, np.nan),
            where=bin_counts > 0,
        )

        ds[new_names[i]] = xr.DataArray(
            mean,
            dims=("x_bin", "y_bin"),
            coords={
                "x_bin": 0.5 * (xedges[:-1] + xedges[1:]),
                "y_bin": 0.5 * (yedges[:-1] + yedges[1:]),
            },
        )
    return ds

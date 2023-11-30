"""
This module provides a function to easily and efficiently plot trajectories stored in a ragged array.
"""

from clouddrift.ragged import segment, rowsize_to_index
import numpy as np
import pandas as pd
from typing import Optional, Union
import xarray as xr
import pandas as pd
from typing import Optional, Union
from clouddrift.ragged import segment, rowsize_to_index


def plot_ragged(
    ax,
    longitude: Union[list, np.ndarray, pd.Series, xr.DataArray],
    latitude: Union[list, np.ndarray, pd.Series, xr.DataArray],
    rowsize: Union[list, np.ndarray, pd.Series, xr.DataArray],
    *args,
    colors: Optional[Union[list, np.ndarray, pd.Series, xr.DataArray]] = None,
    tolerance: Optional[Union[float, int]] = 180,
    **kwargs,
):
    """Plot trajectories from a ragged array dataset on a Matplotlib Axes
    or a Cartopy GeoAxes object ``ax``.

    This function wraps Matplotlib's ``plot`` function (``plt.plot``) and
    ``LineCollection`` (``matplotlib.collections``) to efficiently plot
    trajectories from a ragged array dataset.

    Parameters
    ----------
    ax: matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Axis to plot on.
    longitude : array-like
        Longitude sequence. Unidimensional array input.
    latitude : array-like
        Latitude sequence. Unidimensional array input.
    rowsize : list
        List of integers specifying the number of data points in each row.
    *args : tuple
        Additional arguments to pass to ``ax.plot``.
    colors : array-like
        Colors to use for plotting. If colors is the same shape as longitude and latitude,
        the trajectories are splitted into segments and each segment is colored according
        to the corresponding color value. If colors is the same shape as rowsize, the
        trajectories are uniformly colored according to the corresponding color value.
    tolerance : float
        Longitude tolerance gap between data points (in degrees) for segmenting trajectories.
        For periodic domains, the tolerance parameter should be set to the maximum allowed gap
        between data points. Defaults to 180.
    **kwargs : dict
        Additional keyword arguments to pass to ``ax.plot``.

    Returns
    -------
    list of matplotlib.lines.Line2D or matplotlib.collections.LineCollection
        The plotted lines or line collection. Can be used to set a colorbar
        after plotting or extract information from the lines.

    Examples
    --------

    Plot the first 100 trajectories from the gdp1h dataset, assigning
    a different color to each trajectory:

    >>> from clouddrift import datasets
    >>> import matplotlib.pyplot as plt
    >>> ds = datasets.gdp1h()
    >>> ds = subset(ds, {"ID": ds.ID[:100].values}).load()
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)

    >>> plot_ragged(
    >>>     ax,
    >>>     ds.lon,
    >>>     ds.lat,
    >>>     ds.rowsize,
    >>>     colors=np.arange(len(ds.rowsize))
    >>> )

    To plot the same trajectories, but assigning a different color to each
    observation and specifying a colormap:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> time = [v.astype(np.int64) / 86400 / 1e9 for v in ds.time.values]
    >>> lc = plot_ragged(
    >>>     ax,
    >>>     ds.lon,
    >>>     ds.lat,
    >>>     ds.rowsize,
    >>>     colors=np.floor(time),
    >>>     cmap="inferno"
    >>> )
    >>> fig.colorbar(lc[0])
    >>> ax.set_xlim([-180, 180])
    >>> ax.set_ylim([-90, 90])

    Finally, to plot the same trajectories, but using a cartopy
    projection:

    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    >>> time = [v.astype(np.int64) / 86400 / 1e9 for v in ds.time.values]
    >>> lc = plot_ragged(
    >>>     ax,
    >>>     ds.lon,
    >>>     ds.lat,
    >>>     ds.rowsize,
    >>>     colors=np.arange(len(ds.rowsize)),
    >>>     transform=ccrs.PlateCarree(),
    >>>     cmap=cmocean.cm.ice,
    >>> )

    Raises
    ------
    ValueError
        If longitude and latitude arrays do not have the same shape.
        If colors do not have the same shape as longitude and latitude arrays or rowsize.
        If ax is not a matplotlib Axes or GeoAxes object.
        If ax is a GeoAxes object and the transform keyword argument is not provided.

    ImportError
        If matplotlib is not installed.
        If the axis is a GeoAxes object and cartopy is not installed.
    """

    # optional dependency
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.collections import LineCollection
        from matplotlib import cm
    except ImportError:
        raise ImportError("missing optional dependency 'matplotlib'")

    if hasattr(ax, "coastlines"):  # check if GeoAxes without cartopy
        try:
            from cartopy.mpl.geoaxes import GeoAxes

            if isinstance(ax, GeoAxes) and not kwargs.get("transform"):
                raise ValueError(
                    "For GeoAxes, the transform keyword argument must be provided."
                )
        except ImportError:
            raise ImportError("missing optional dependency 'cartopy'")
    elif not isinstance(ax, plt.Axes):
        raise ValueError("ax must be either: plt.Axes or GeoAxes.")

    if np.sum(rowsize) != len(longitude):
        raise ValueError("The sum of rowsize must equal the length of lon and lat.")

    if len(longitude) != len(latitude):
        raise ValueError("lon and lat must have the same length.")

    if colors is None:
        colors = np.arange(len(rowsize))
    elif colors is not None and (len(colors) not in [len(longitude), len(rowsize)]):
        raise ValueError("shape colors must match the shape of lon/lat or rowsize.")

    # define a colormap
    cmap = kwargs.pop("cmap", cm.viridis)

    # define a normalization obtain uniform colors
    # for the sequence of lines or LineCollection
    norm = kwargs.pop(
        "norm", mcolors.Normalize(vmin=np.nanmin(colors), vmax=np.nanmax(colors))
    )

    mpl_plot = True if colors is None or len(colors) == len(rowsize) else False
    traj_idx = rowsize_to_index(rowsize)

    lines = []
    for i in range(len(rowsize)):
        lon_i, lat_i = (
            longitude[traj_idx[i] : traj_idx[i + 1]],
            latitude[traj_idx[i] : traj_idx[i + 1]],
        )

        start = 0
        for length in segment(lon_i, tolerance, rowsize=segment(lon_i, -tolerance)):
            end = start + length

            if mpl_plot:
                line = ax.plot(
                    lon_i[start:end],
                    lat_i[start:end],
                    c=cmap(norm(colors[i])) if colors is not None else None,
                    *args,
                    **kwargs,
                )
            else:
                colors_i = colors[traj_idx[i] : traj_idx[i + 1]]
                segments = np.column_stack(
                    [
                        lon_i[start : end - 1],
                        lat_i[start : end - 1],
                        lon_i[start + 1 : end],
                        lat_i[start + 1 : end],
                    ]
                ).reshape(-1, 2, 2)
                line = LineCollection(segments, cmap=cmap, norm=norm, *args, **kwargs)
                line.set_array(
                    # color of a segment is the average of its two data points
                    np.convolve(colors_i[start:end], [0.5, 0.5], mode="valid")
                )
                ax.add_collection(line)

            start = end
            lines.append(line)

    # set axis limits
    ax.set_xlim([np.min(longitude), np.max(longitude)])
    ax.set_ylim([np.min(latitude), np.max(latitude)])

    return lines

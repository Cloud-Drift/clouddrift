"""
This module provides a function to easily and efficiently plot trajectories stored in a ragged array.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.ragged import rowsize_to_index, segment


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

    Load 100 trajectories from the gdp1h dataset for the examples.

    >>> from clouddrift import datasets
    >>> from clouddrift.ragged import subset
    >>> from clouddrift.plotting import plot_ragged
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1 import make_axes_locatable
    >>> ds = datasets.gdp1h()
    >>> ds = subset(ds, {"id": ds.id[:100].values}).load()

    Plot the trajectories, assigning a different color to each trajectory:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> l = plot_ragged(
    >>>     ax,
    >>>     ds.lon,
    >>>     ds.lat,
    >>>     ds.rowsize,
    >>>     colors=np.arange(len(ds.rowsize))
    >>> )
    >>> divider = make_axes_locatable(ax)
    >>> cax = divider.append_axes('right', size='3%', pad=0.05)
    >>> fig.colorbar(l, cax=cax)

    To plot the same trajectories, but assigning a different color to each
    observation based on time and specifying a colormap:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> time = [v.astype(np.int64) / 86400 / 1e9 for v in ds.time.values]
    >>> l = plot_ragged(
    >>>     ax,
    >>>     ds.lon,
    >>>     ds.lat,
    >>>     ds.rowsize,
    >>>     colors=np.floor(time),
    >>>     cmap="inferno"
    >>> )
    >>> divider = make_axes_locatable(ax)
    >>> cax = divider.append_axes('right', size="3%", pad=0.05)
    >>> fig.colorbar(l, cax=cax)

    Finally, to plot the same trajectories, but using a cartopy
    projection:

    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    >>> l = plot_ragged(
    >>>     ax,
    >>>     ds.lon,
    >>>     ds.lat,
    >>>     ds.rowsize,
    >>>     colors=np.arange(len(ds.rowsize)),
    >>>     transform=ccrs.PlateCarree(),
    >>>     cmap="Blues",
    >>> )
    >>> ax.set_extent([-180, 180, -90, 90])
    >>> ax.coastlines()
    >>> ax.gridlines(draw_labels=True)
    >>> divider = make_axes_locatable(ax)
    >>> cax = divider.append_axes('right', size="3%", pad=0.25, axes_class=plt.Axes)
    >>> fig.colorbar(l, cax=cax)

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
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.collections import LineCollection
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
    if isinstance(cmap := kwargs.pop("cmap", cm.viridis), str):
        cmap = plt.get_cmap(cmap)

    # define a normalization obtain uniform colors
    # for the sequence of lines or LineCollection
    norm = kwargs.pop(
        "norm", mcolors.Normalize(vmin=np.nanmin(colors), vmax=np.nanmax(colors))
    )

    # create Mappable for colorbar
    cb = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    mpl_plot = True if colors is None or len(colors) == len(rowsize) else False
    traj_idx = rowsize_to_index(rowsize)

    for i in range(len(rowsize)):
        lon_i, lat_i = (
            longitude[traj_idx[i] : traj_idx[i + 1]],
            latitude[traj_idx[i] : traj_idx[i + 1]],
        )

        start = 0
        for length in segment(lon_i, tolerance, rowsize=segment(lon_i, -tolerance)):
            end = start + length

            if mpl_plot:
                ax.plot(
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
                lc = LineCollection(segments, cmap=cmap, norm=norm, *args, **kwargs)
                lc.set_array(
                    # color of a segment is the average of its two data points
                    np.convolve(colors_i[start:end], [0.5, 0.5], mode="valid")
                )
                ax.add_collection(lc)

            start = end

    # set axis limits
    ax.set_xlim([np.min(longitude), np.max(longitude)])
    ax.set_ylim([np.min(latitude), np.max(latitude)])

    return cb

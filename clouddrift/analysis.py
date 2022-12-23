from datetime import datetime
import numpy as np
from typing import Optional, Tuple
import xarray as xr


def velocity_from_positions(
    x: xr.DataArray,
    y: xr.DataArray,
    time: xr.DataArray,
    coord_system: Optional[str] = "spherical",
    difference_scheme: Optional[str] = "forward",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute velocity in meters per second given arrays of positions and
    time.

    x and y can be provided as longitude and latitude in degrees if
    coord_system == "spherical" (default), or as northing and easting in meters
    if coord_system == "cartesian".

    Difference scheme can take one of three values:

        1. "forward" (default): finite difference is evaluated as
           dx[i] = dx[i+1] - dx[i];
        2. "backward": finite difference is evaluated as
           dx[i] = dx[i] - dx[i-1];
        3. "centered": finite difference is evaluated as
           dx[i] = (dx[i+1] - dx[i-1]) / 2.

    Forward and backward schemes are effectively the same except that the
    position at which the velocity is evaluated is shifted one element down in
    the backward scheme relative to the forward scheme. In the case of a
    forward or backward difference scheme, the last or first element of the
    velocity, respectively, is extrapolated from its neighboring point. In the
    case of a centered difference scheme, the start and end boundary points are
    evaluated using the forward and backward difference scheme, respectively.

    Args:
        x (xr.DataArray[float]): An array of x-positions (longitude in degrees or easting in meters)
        y (xr.DataArray[float]): An array of y-positions (latitude in degrees or northing in meters)
        time (xr.DataArray[datetime]): An array of times
        coord_system (str, optional): Coordinate system that x and y arrays are in; possible values are "spherical" (default) or "cartesian".
        difference_scheme (str, optional): Difference scheme to use; possible values are "forward", "backward", and "centered".

    Returns:
        out (Tuple[xr.DataArray[float], xr.DataArray[float]]): Arrays of x- and y-velocities in meters per second
    """

    # Positions and time arrays must have the same shape.
    if not x.shape == y.shape == time.shape:
        raise ValueError("x, y, and time must have the same shape.")

    #FIXME DataArray construction
    dx = xr.DataArray(x.shape)
    dy = xr.DataArray(y.shape)
    dt = xr.DataArray(time.shape)

    # Compute dx, dy, and dt
    if difference_scheme == "forward":

        # Time
        dt[:-1] = np.diff(time)
        dt[-1] = dt[-2]

        # Space
        if coord_system == "cartesian":
            dx[:-1] = np.diff(x)
            dx[-1] = dx[-2]
            dy[:-1] = np.diff(y)
            dy[-1] = dy[-2]
        elif coord_system == "spherical":
            # TODO compute Haversine (distance, bearing) -> (dx, dy)
            raise NotImplementedError()
        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    elif difference_scheme == "backward":

        # Time
        dt[1:] = np.diff(time)
        dt[0] = dt[1]

        # Space
        if coord_system == "cartesian":
            dx[1:] = np.diff(x)
            dx[0] = dx[1]
            dy[1:] = np.diff(y)
            dy[0] = dy[1]
        elif coord_system == "spherical":
            # TODO compute Haversine (distance, bearing) -> (dx, dy)
            raise NotImplementedError()
        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    elif difference_scheme == "centered":

        # Time
        dt[1:-1] = (time[2:] - time[:-2]) / 2
        dt[0] = time[1] - time[0]
        dt[-1] = time[-1] - time[-2]

        # Space
        if coord_system == "cartesian":
            dx[1:-1] = (x[2:] - x[:-2]) / 2
            dx[0] = x[1] - x[0]
            dx[-1] = x[-1] - x[-2]
            dy[1:-1] = (y[2:] - y[:-2]) / 2
            dy[0] = y[1] - y[0]
            dy[-1] = y[-1] - y[-2]
        elif coord_system == "spherical":
            # TODO compute Haversine (distance, bearing) -> (dx, dy)
            raise NotImplementedError()
        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    else:
        raise ValueError(
            'difference_scheme must be "forward", "backward", or "centered".'
        )

    return dx / dt, dy / dt

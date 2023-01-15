import numpy as np
from typing import Optional, Tuple
import xarray as xr
from clouddrift.haversine import distance, bearing


def velocity_from_position(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    coord_system: Optional[str] = "spherical",
    difference_scheme: Optional[str] = "forward",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute velocity from arrays of positions and time.

    x and y can be provided as longitude and latitude in degrees if
    coord_system == "spherical" (default), or as easting and northing if
    coord_system == "cartesian".

    The units of the result are meters per unit of time if
    coord_system == "spherical". For example, if the time is provided in the
    units of seconds, the resulting velocity is in the units of meters per
    second. Otherwise, if coord_system == "cartesian", the units of the
    resulting velocity correspond to the units of the input. For example,
    if Easting and Northing are in the units of kilometers and time is in
    the units of hours, the resulting velocity is in the units of kilometers
    per hour.

    x, y, and time can be multi-dimensional arrays, but the last (fastest-varying)
    dimension must be the time dimension along which the differencing is done.
    x, y, and time must have the same shape.

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
        x (array_like): An N-d array of x-positions (longitude in degrees or easting in any unit)
            where the last (fastest-varying) dimension is the time
        y (array_like): An N-d array of y-positions (latitude in degrees or northing in any unit)
            where the last (fastest-varying) dimension is the time
        time (array_like): An N-d array of times as floating point values (in any unit)
            where the last (fastest-varying) dimension is the time
        coord_system (str, optional): Coordinate system that x and y arrays are in; possible values are "spherical" (default) or "cartesian".
        difference_scheme (str, optional): Difference scheme to use; possible values are "forward", "backward", and "centered".

    Returns:
        out (Tuple[xr.DataArray[float], xr.DataArray[float]]): Arrays of x- and y-velocities
    """

    # Positions and time arrays must have the same shape.
    if not x.shape == y.shape == time.shape:
        raise ValueError("x, y, and time must have the same shape.")

    dx = np.empty(x.shape)
    dy = np.empty(y.shape)
    dt = np.empty(time.shape)

    # Compute dx, dy, and dt
    if difference_scheme == "forward":

        # All values except the ending boundary value are computed using the
        # 1st order forward differencing. The ending boundary value is
        # computed using the 1st order backward difference.

        # Time
        dt[..., :-1] = np.diff(time)
        dt[..., -1] = dt[..., -2]

        # Space
        if coord_system == "cartesian":

            dx[..., :-1] = np.diff(x)
            dx[..., -1] = dx[..., -2]
            dy[..., :-1] = np.diff(y)
            dy[..., -1] = dy[..., -2]

        elif coord_system == "spherical":

            distances = distance(y[..., :-1], x[..., :-1], y[..., 1:], x[..., 1:])
            bearings = bearing(y[..., :-1], x[..., :-1], y[..., 1:], x[..., 1:])
            dx[..., :-1] = distances * np.cos(bearings)
            dx[..., -1] = dx[..., -2]
            dy[..., :-1] = distances * np.sin(bearings)
            dy[..., -1] = dy[..., -2]

        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    elif difference_scheme == "backward":

        # All values except the starting boundary value are computed using the
        # 1st order backward differencing. The starting boundary value is
        # computed using the 1st order forward difference.

        # Time
        dt[..., 1:] = np.diff(time)
        dt[..., 0] = dt[..., 1]

        # Space
        if coord_system == "cartesian":

            dx[..., 1:] = np.diff(x)
            dx[..., 0] = dx[..., 1]
            dy[..., 1:] = np.diff(y)
            dy[..., 0] = dy[..., 1]

        elif coord_system == "spherical":

            distances = distance(y[..., :-1], x[..., :-1], y[..., 1:], x[..., 1:])
            bearings = bearing(y[..., :-1], x[..., :-1], y[..., 1:], x[..., 1:])
            dx[..., 1:] = distances * np.cos(bearings)
            dx[..., 0] = dx[..., 1]
            dy[..., 1:] = distances * np.sin(bearings)
            dy[..., 0] = dy[..., 1]

        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    elif difference_scheme == "centered":

        # Inner values are computed using the 2nd order centered differencing.
        # The start and end boundary values are computed using the 1st order
        # forward and backward differencing, respectively.

        # Time
        dt[..., 1:-1] = (time[..., 2:] - time[..., :-2]) / 2
        dt[..., 0] = time[..., 1] - time[..., 0]
        dt[..., -1] = time[..., -1] - time[..., -2]

        # Space
        if coord_system == "cartesian":

            dx[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / 2
            dx[..., 0] = x[..., 1] - x[..., 0]
            dx[..., -1] = x[..., -1] - x[..., -2]
            dy[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / 2
            dy[..., 0] = y[..., 1] - y[..., 0]
            dy[..., -1] = y[..., -1] - y[..., -2]

        elif coord_system == "spherical":

            # Inner values
            distances = distance(y[..., :-2], x[..., :-2], y[..., 2:], x[..., 2:])
            bearings = bearing(y[..., :-2], x[..., :-2], y[..., 2:], x[..., 2:])
            dx[..., 1:-1] = distances * np.cos(bearings) / 2
            dy[..., 1:-1] = distances * np.sin(bearings) / 2

            # Boundary values
            distance1 = distance(y[..., 0], x[..., 0], y[..., 1], x[..., 1])
            bearing1 = bearing(y[..., 0], x[..., 0], y[..., 1], x[..., 1])
            dx[..., 0] = distance1 * np.cos(bearing1)
            dy[..., 0] = distance1 * np.sin(bearing1)
            distance2 = distance(y[..., -2], x[..., -2], y[..., -1], x[..., -1])
            bearing2 = bearing(y[..., -2], x[..., -2], y[..., -1], x[..., -1])
            dx[..., -1] = distance2 * np.cos(bearing2)
            dy[..., -1] = distance2 * np.sin(bearing2)

        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    else:
        raise ValueError(
            'difference_scheme must be "forward", "backward", or "centered".'
        )

    return dx / dt, dy / dt

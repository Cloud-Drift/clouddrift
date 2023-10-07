"""
Functions for kinematic computations.
"""

import numpy as np
from typing import Optional, Tuple
import xarray as xr
from clouddrift.sphere import distance, bearing, position_from_distance_and_bearing


def position_from_velocity(
    u: np.ndarray,
    v: np.ndarray,
    time: np.ndarray,
    x_origin: float,
    y_origin: float,
    coord_system: Optional[str] = "spherical",
    integration_scheme: Optional[str] = "forward",
    time_axis: Optional[int] = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positions from arrays of velocities and time and a pair of origin
    coordinates.

    The units of the result are degrees if ``coord_system == "spherical"`` (default).
    If ``coord_system == "cartesian"``, the units of the result are equal to the
    units of the input velocities multiplied by the units of the input time.
    For example, if the input velocities are in meters per second and the input
    time is in seconds, the units of the result will be meters.

    Integration scheme can take one of three values:

        1. "forward" (default): integration from x[i] to x[i+1] is performed
            using the velocity at x[i].
        2. "backward": integration from x[i] to x[i+1] is performed using the
            velocity at x[i+1].
        3. "centered": integration from x[i] to x[i+1] is performed using the
            arithmetic average of the velocities at x[i] and x[i+1]. Note that
            this method introduces some error due to the averaging.

    u, v, and time can be multi-dimensional arrays. If the time axis, along
    which the finite differencing is performed, is not the last one (i.e.
    x.shape[-1]), use the ``time_axis`` optional argument to specify along which
    axis should the differencing be done. ``x``, ``y``, and ``time`` must have
    the same shape.

    This function will not do any special handling of longitude ranges. If the
    integrated trajectory crosses the antimeridian (dateline) in either direction, the
    longitude values will not be adjusted to stay in any specific range such
    as [-180, 180] or [0, 360]. If you need your longitudes to be in a specific
    range, recast the resulting longitude from this function using the function
    :func:`clouddrift.sphere.recast_lon`.

    Parameters
    ----------
    u : np.ndarray
        An array of eastward velocities.
    v : np.ndarray
        An array of northward velocities.
    time : np.ndarray
        An array of time values.
    x_origin : float
        Origin x-coordinate or origin longitude.
    y_origin : float
        Origin y-coordinate or origin latitude.
    coord_system : str, optional
        The coordinate system of the input. Can be "spherical" or "cartesian".
        Default is "spherical".
    integration_scheme : str, optional
        The difference scheme to use for computing the position. Can be
        "forward" or "backward". Default is "forward".
    time_axis : int, optional
        The axis of the time array. Default is -1, which corresponds to the
        last axis.

    Returns
    -------
    x : np.ndarray
        An array of zonal displacements or longitudes.
    y : np.ndarray
        An array of meridional displacements or latitudes.

    Examples
    --------

    Simple integration on a plane, using the forward scheme by default:

    >>> import numpy as np
    >>> from clouddrift.analysis import position_from_velocity
    >>> u = np.array([1., 2., 3., 4.])
    >>> v = np.array([1., 1., 1., 1.])
    >>> time = np.array([0., 1., 2., 3.])
    >>> x, y = position_from_velocity(u, v, time, 0, 0, coord_system="cartesian")
    >>> x
    array([0., 1., 3., 6.])
    >>> y
    array([0., 1., 2., 3.])

    As above, but using centered scheme:

    >>> x, y = position_from_velocity(u, v, time, 0, 0, coord_system="cartesian", integration_scheme="centered")
    >>> x
    array([0., 1.5, 4., 7.5])
    >>> y
    array([0., 1., 2., 3.])

    Simple integration on a sphere (default):

    >>> u = np.array([1., 2., 3., 4.])
    >>> v = np.array([1., 1., 1., 1.])
    >>> time = np.array([0., 1., 2., 3.]) * 1e5
    >>> x, y = position_from_velocity(u, v, time, 0, 0)
    >>> x
    array([0.        , 0.89839411, 2.69584476, 5.39367518])
    >>> y
    array([0.        , 0.89828369, 1.79601515, 2.69201609])

    Integrating across the antimeridian (dateline) by default does not
    recast the resulting longitude:

    >>> u = np.array([1., 1.])
    >>> v = np.array([0., 0.])
    >>> time = np.array([0, 1e5])
    >>> x, y = position_from_velocity(u, v, time, 179.5, 0)
    >>> x
    array([179.5      , 180.3983205])
    >>> y
    array([0., 0.])

    Use the ``clouddrift.sphere.recast_lon`` function to recast the longitudes
    to the desired range:

    >>> from clouddrift.sphere import recast_lon
    >>> recast_lon(x, -180)
    array([ 179.5      , -179.6016795])

    Raises
    ------
    ValueError
        If u and v do not have the same shape.
        If the time axis is outside of the valid range ([-1, N-1]).
        If lengths of x, y, and time along time_axis are not equal.
        If the input coordinate system is not "spherical" or "cartesian".
        If the input integration scheme is not "forward", "backward", or "centered"

    See Also
    --------
    :func:`velocity_from_position`
    """
    # Velocity arrays must have the same shape.
    # Although the exception would be raised further down in the function,
    # we do the check here for a clearer error message.
    if not u.shape == u.shape:
        raise ValueError("u and v must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(u.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )

    # Input arrays must have the same length along the time axis.
    if not u.shape[time_axis] == v.shape[time_axis] == time.shape[time_axis]:
        raise ValueError(
            f"u, v, and time must have the same length along the time axis "
            f"({time_axis})."
        )

    # Swap axes so that we can differentiate along the last axis.
    # This is a syntax convenience rather than memory access optimization:
    # np.swapaxes returns a view of the array, not a copy, if the input is a
    # NumPy array. Otherwise, it returns a copy. For readability, introduce new
    # variable names so that we can more easily differentiate between the
    # original arrays and those with swapped axes.
    u_ = np.swapaxes(u, time_axis, -1)
    v_ = np.swapaxes(v, time_axis, -1)
    time_ = np.swapaxes(time, time_axis, -1)

    x = np.zeros(u_.shape, dtype=u.dtype)
    y = np.zeros(v_.shape, dtype=v.dtype)

    dt = np.diff(time_)

    if integration_scheme.lower() == "forward":
        x[..., 1:] = np.cumsum(u_[..., :-1] * dt, axis=-1)
        y[..., 1:] = np.cumsum(v_[..., :-1] * dt, axis=-1)
    elif integration_scheme.lower() == "backward":
        x[..., 1:] = np.cumsum(u_[1:] * dt, axis=-1)
        y[..., 1:] = np.cumsum(v_[1:] * dt, axis=-1)
    elif integration_scheme.lower() == "centered":
        x[..., 1:] = np.cumsum(0.5 * (u_[..., :-1] + u_[..., 1:]) * dt, axis=-1)
        y[..., 1:] = np.cumsum(0.5 * (v_[..., :-1] + v_[..., 1:]) * dt, axis=-1)
    else:
        raise ValueError(
            'integration_scheme must be "forward", "backward", or "centered".'
        )

    if coord_system.lower() == "cartesian":
        x += x_origin
        y += y_origin
    elif coord_system.lower() == "spherical":
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        bearings = np.arctan2(dy, dx)
        x[..., 0], y[..., 0] = x_origin, y_origin
        for n in range(distances.shape[-1]):
            y[..., n + 1], x[..., n + 1] = position_from_distance_and_bearing(
                y[..., n], x[..., n], distances[..., n], bearings[..., n]
            )
    else:
        raise ValueError('coord_system must be "spherical" or "cartesian".')

    return np.swapaxes(x, time_axis, -1), np.swapaxes(y, time_axis, -1)


def velocity_from_position(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    coord_system: Optional[str] = "spherical",
    difference_scheme: Optional[str] = "forward",
    time_axis: Optional[int] = -1,
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
    if zonal and meridional displacements are in the units of kilometers and
    time is in the units of hours, the resulting velocity is in the units of
    kilometers per hour.

    x, y, and time can be multi-dimensional arrays. If the time axis, along
    which the finite differencing is performed, is not the last one (i.e.
    x.shape[-1]), use the time_axis optional argument to specify along which
    axis should the differencing be done. x, y, and time must have the same
    shape.

    Difference scheme can take one of three values:

    #. "forward" (default): finite difference is evaluated as ``dx[i] = dx[i+1] - dx[i]``;
    #. "backward": finite difference is evaluated as ``dx[i] = dx[i] - dx[i-1]``;
    #. "centered": finite difference is evaluated as ``dx[i] = (dx[i+1] - dx[i-1]) / 2``.

    Forward and backward schemes are effectively the same except that the
    position at which the velocity is evaluated is shifted one element down in
    the backward scheme relative to the forward scheme. In the case of a
    forward or backward difference scheme, the last or first element of the
    velocity, respectively, is extrapolated from its neighboring point. In the
    case of a centered difference scheme, the start and end boundary points are
    evaluated using the forward and backward difference scheme, respectively.

    Parameters
    ----------
    x : array_like
        An N-d array of x-positions (longitude in degrees or zonal displacement in any unit)
    y : array_like
        An N-d array of y-positions (latitude in degrees or meridional displacement in any unit)
    time : array_like
        An N-d array of times as floating point values (in any unit)
    coord_system : str, optional
        Coordinate system that x and y arrays are in; possible values are "spherical" (default) or "cartesian".
    difference_scheme : str, optional
        Difference scheme to use; possible values are "forward", "backward", and "centered".
    time_axis : int, optional
        Axis along which to differentiate (default is -1)

    Returns
    -------
    u : np.ndarray
        Zonal velocity
    v : np.ndarray
        Meridional velocity

    Raises
    ------
    ValueError
        If x and y do not have the same shape.
        If time_axis is outside of the valid range.
        If lengths of x, y, and time along time_axis are not equal.
        If coord_system is not "spherical" or "cartesian".
        If difference_scheme is not "forward", "backward", or "centered".

    See Also
    --------
    :func:`position_from_velocity`
    """

    # Position arrays must have the same shape.
    # Although the exception would be raised further down in the function,
    # we do the check here for a clearer error message.
    if not x.shape == y.shape:
        raise ValueError("x and y arrays must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(x.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )

    # Input arrays must have the same length along the time axis.
    if not x.shape[time_axis] == y.shape[time_axis] == time.shape[time_axis]:
        raise ValueError(
            f"x, y, and time must have the same length along the time axis "
            f"({time_axis})."
        )

    # Swap axes so that we can differentiate along the last axis.
    # This is a syntax convenience rather than memory access optimization:
    # np.swapaxes returns a view of the array, not a copy, if the input is a
    # NumPy array. Otherwise, it returns a copy. For readability, introduce new
    # variable names so that we can more easily differentiate between the
    # original arrays and those with swapped axes.
    x_ = np.swapaxes(x, time_axis, -1)
    y_ = np.swapaxes(y, time_axis, -1)
    time_ = np.swapaxes(time, time_axis, -1)

    dx = np.empty(x_.shape)
    dy = np.empty(y_.shape)
    dt = np.empty(time_.shape)

    # Compute dx, dy, and dt
    if difference_scheme == "forward":
        # All values except the ending boundary value are computed using the
        # 1st order forward differencing. The ending boundary value is
        # computed using the 1st order backward difference.

        # Time
        dt[..., :-1] = np.diff(time_)
        dt[..., -1] = dt[..., -2]

        # Space
        if coord_system == "cartesian":
            dx[..., :-1] = np.diff(x_)
            dx[..., -1] = dx[..., -2]
            dy[..., :-1] = np.diff(y_)
            dy[..., -1] = dy[..., -2]

        elif coord_system == "spherical":
            distances = distance(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
            bearings = bearing(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
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
        dt[..., 1:] = np.diff(time_)
        dt[..., 0] = dt[..., 1]

        # Space
        if coord_system == "cartesian":
            dx[..., 1:] = np.diff(x_)
            dx[..., 0] = dx[..., 1]
            dy[..., 1:] = np.diff(y_)
            dy[..., 0] = dy[..., 1]

        elif coord_system == "spherical":
            distances = distance(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
            bearings = bearing(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
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
        dt[..., 1:-1] = (time_[..., 2:] - time_[..., :-2]) / 2
        dt[..., 0] = time_[..., 1] - time_[..., 0]
        dt[..., -1] = time_[..., -1] - time_[..., -2]

        # Space
        if coord_system == "cartesian":
            dx[..., 1:-1] = (x_[..., 2:] - x_[..., :-2]) / 2
            dx[..., 0] = x_[..., 1] - x_[..., 0]
            dx[..., -1] = x_[..., -1] - x_[..., -2]
            dy[..., 1:-1] = (y_[..., 2:] - y_[..., :-2]) / 2
            dy[..., 0] = y_[..., 1] - y_[..., 0]
            dy[..., -1] = y_[..., -1] - y_[..., -2]

        elif coord_system == "spherical":
            # Inner values
            y1 = (y_[..., :-2] + y_[..., 1:-1]) / 2
            x1 = (x_[..., :-2] + x_[..., 1:-1]) / 2
            y2 = (y_[..., 2:] + y_[..., 1:-1]) / 2
            x2 = (x_[..., 2:] + x_[..., 1:-1]) / 2
            distances = distance(y1, x1, y2, x2)
            bearings = bearing(y1, x1, y2, x2)
            dx[..., 1:-1] = distances * np.cos(bearings)
            dy[..., 1:-1] = distances * np.sin(bearings)

            # Boundary values
            distance1 = distance(y_[..., 0], x_[..., 0], y_[..., 1], x_[..., 1])
            bearing1 = bearing(y_[..., 0], x_[..., 0], y_[..., 1], x_[..., 1])
            dx[..., 0] = distance1 * np.cos(bearing1)
            dy[..., 0] = distance1 * np.sin(bearing1)
            distance2 = distance(y_[..., -2], x_[..., -2], y_[..., -1], x_[..., -1])
            bearing2 = bearing(y_[..., -2], x_[..., -2], y_[..., -1], x_[..., -1])
            dx[..., -1] = distance2 * np.cos(bearing2)
            dy[..., -1] = distance2 * np.sin(bearing2)

        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    else:
        raise ValueError(
            'difference_scheme must be "forward", "backward", or "centered".'
        )

    # This should avoid an array copy when returning the result
    dx /= dt
    dy /= dt

    return np.swapaxes(dx, time_axis, -1), np.swapaxes(dy, time_axis, -1)

"""
Functions for kinematic computations.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.sphere import (
    EARTH_RADIUS_METERS,
    bearing,
    cartesian_to_spherical,
    cartesian_to_tangentplane,
    coriolis_frequency,
    distance,
    position_from_distance_and_bearing,
    recast_lon360,
    spherical_to_cartesian,
)
from clouddrift.wavelet import morse_logspace_freq, morse_wavelet, wavelet_transform


def kinetic_energy(
    u: Union[float, list, np.ndarray, xr.DataArray, pd.Series],
    v: Optional[Union[float, list, np.ndarray, xr.DataArray, pd.Series]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Compute kinetic energy from zonal and meridional velocities.

    Parameters
    ----------
    u : float or array-like
        Zonal velocity.
    v : float or array-like, optional.
        Meridional velocity. If not provided, the flow is assumed one-dimensional
        in time and defined by ``u``.

    Returns
    -------
    ke : float or array-like
        Kinetic energy.

    Examples
    --------
    >>> import numpy as np
    >>> from clouddrift.kinematics import kinetic_energy
    >>> u = np.array([1., 2., 3., 4.])
    >>> v = np.array([1., 1., 1., 1.])
    >>> kinetic_energy(u, v)
    array([1. , 2.5, 5. , 8.5])

    >>> u = np.reshape(np.tile([1., 2., 3., 4.], 2), (2, 4))
    >>> v = np.reshape(np.tile([1., 1., 1., 1.], 2), (2, 4))
    >>> kinetic_energy(u, v)
    array([[1. , 2.5, 5. , 8.5],
           [1. , 2.5, 5. , 8.5]])
    """
    if v is None:
        v = np.zeros_like(u)
    ke = (u**2 + v**2) / 2
    return ke


def inertial_oscillation_from_position(
    longitude: np.ndarray,
    latitude: np.ndarray,
    relative_bandwidth: Optional[float] = None,
    wavelet_duration: Optional[float] = None,
    time_step: Optional[float] = 3600.0,
    relative_vorticity: Optional[Union[float, np.ndarray]] = 0.0,
) -> np.ndarray:
    """Extract inertial oscillations from consecutive geographical positions.

    This function acts by performing a time-frequency analysis of horizontal displacements
    with analytic Morse wavelets. It extracts the portion of the wavelet transform signal
    that follows the inertial frequency (opposite of Coriolis frequency) as a function of time,
    potentially shifted in frequency by a measure of relative vorticity. The result is a pair
    of zonal and meridional relative displacements in meters.

    This function is equivalent to a bandpass filtering of the horizontal displacements. The characteristics
    of the filter are defined by the relative bandwidth of the wavelet transform or by the duration of the wavelet,
    see the parameters below.

    Parameters
    ----------
    longitude : array-like
        Longitude sequence. Unidimensional array input.
    latitude : array-like
        Latitude sequence. Unidimensional array input.
    relative_bandwidth : float, optional
        Bandwidth of the frequency-domain equivalent filter for the extraction of the inertial
        oscillations; a number less or equal to one which is a fraction of the inertial frequency.
        A value of 0.1 leads to a bandpass filter equivalent of +/- 10 percent of the inertial frequency.
    wavelet_duration : float, optional
        Duration of the wavelet, or inverse of the relative bandwidth, which can be passed instead of the
        relative bandwidth.
    time_step : float, optional
        The constant time interval between data points in seconds. Default is 3600.
    relative_vorticity: Optional, float or array-like
        Relative vorticity adding to the local Coriolis frequency. If "f" is the Coriolis
        frequency then "f" + `relative_vorticity` will be the effective Coriolis frequency as defined by Kunze (1985).
        Positive values correspond to cyclonic vorticity, irrespectively of the latitudes of the data
        points.

    Returns
    -------
    xhat : array-like
        Zonal relative displacement in meters from inertial oscillations.
    yhat : array-like
        Meridional relative displacement in meters from inertial oscillations.

    Examples
    --------
    To extract displacements from inertial oscillations from sequences of longitude
    and latitude values, equivalent to bandpass around 20 percent of the local inertial frequency:

    >>> xhat, yhat = inertial_oscillation_from_position(longitude, latitude, relative_bandwidth=0.2)

    The same result can be obtained by specifying the wavelet duration instead of the relative bandwidth:

    >>> xhat, yhat = inertial_oscillation_from_position(longitude, latitude, wavelet_duration=5)

    Next, the residual positions from the inertial displacements can be obtained with another function:

    >>> residual_longitudes, residual_latitudes = residual_position_from_displacement(longitude, latitude, xhat, yhat)

    Raises
    ------
    ValueError
        If longitude and latitude arrays do not have the same shape.
        If relative_vorticity is an array and does not have the same shape as longitude and latitude.
        If time_step is not a float.
        If both relative_bandwidth and wavelet_duration are specified.
        If neither relative_bandwidth nor wavelet_duration are specified.
        If the absolute value of relative_bandwidth is not in the range (0,1].
        If the wavelet duration is not greater than or equal to 1.

    See Also
    --------
    :func:`residual_position_from_displacement`, `wavelet_transform`, `morse_wavelet`

    """
    if longitude.shape != latitude.shape:
        raise ValueError("longitude and latitude arrays must have the same shape.")

    if relative_bandwidth is not None and wavelet_duration is not None:
        raise ValueError(
            "Only one of 'relative_bandwidth' and 'wavelet_duration' can be specified"
        )
    elif relative_bandwidth is None and wavelet_duration is None:
        raise ValueError(
            "One of 'relative_bandwidth' and 'wavelet_duration' must be specified"
        )

    # length of data sequence
    data_length = longitude.shape[0]

    if isinstance(relative_vorticity, float):
        relative_vorticity = np.full_like(longitude, relative_vorticity)
    elif isinstance(relative_vorticity, np.ndarray):
        if not relative_vorticity.shape == longitude.shape:
            raise ValueError(
                "relative_vorticity must be a float or the same shape as longitude and latitude."
            )
    if relative_bandwidth is not None:
        if not 0 < np.abs(relative_bandwidth) <= 1:
            raise ValueError("relative_bandwidth must be in the (0, 1]) range")

    if wavelet_duration is not None:
        if not wavelet_duration >= 1:
            raise ValueError("wavelet_duration must be greater than or equal to 1")

    # wavelet parameters are gamma and beta
    gamma = 3  # symmetric wavelet
    density = 16  # results relative insensitive to this parameter
    # calculate beta from wavelet duration or from relative bandwidth
    if relative_bandwidth is not None:
        wavelet_duration = 1 / np.abs(relative_bandwidth)  # P parameter
    beta = wavelet_duration**2 / gamma

    if isinstance(latitude, xr.DataArray):
        latitude = latitude.to_numpy()
    if isinstance(longitude, xr.DataArray):
        longitude = longitude.to_numpy()

    # Instantaneous absolute frequency of oscillations along trajectory in radian per second
    cor_freq = np.abs(
        coriolis_frequency(latitude) + relative_vorticity * np.sign(latitude)
    )
    cor_freq_max = np.max(cor_freq * 1.05)
    cor_freq_min = np.max(
        [np.min(cor_freq * 0.95), 2 * np.pi / (time_step * data_length)]
    )

    # logarithmically distributed frequencies for wavelet analysis
    radian_frequency = morse_logspace_freq(
        gamma,
        beta,
        data_length,
        (0.05, cor_freq_max * time_step),
        (5, cor_freq_min * time_step),
        density,
    )  # frequencies in radian per unit time

    # wavelet transform on a sphere
    # unwrap longitude recasted in [0,360)
    longitude_unwrapped = np.unwrap(recast_lon360(longitude), period=360)

    # convert lat/lon to Cartesian coordinates x, y , z
    x, y, z = spherical_to_cartesian(longitude_unwrapped, latitude)

    # wavelet transform of x, y, z
    wavelet, _ = morse_wavelet(data_length, gamma, beta, radian_frequency)
    wx = wavelet_transform(x, wavelet, boundary="mirror")
    wy = wavelet_transform(y, wavelet, boundary="mirror")
    wz = wavelet_transform(z, wavelet, boundary="mirror")

    longitude_new, latitude_new = cartesian_to_spherical(
        x - np.real(wx), y - np.real(wy), z - np.real(wz)
    )

    # convert transforms to horizontal displacements on tangent plane
    wxh, wyh = cartesian_to_tangentplane(wx, wy, wz, longitude_new, latitude_new)

    # rotary wavelet transforms to select inertial component; need to divide by sqrt(2)
    wp = (wxh + 1j * wyh) / np.sqrt(2)
    wn = (wxh - 1j * wyh) / np.sqrt(2)

    # find the values of radian_frequency/dt that most closely match cor_freq
    frequency_bins = [
        np.argmin(np.abs(cor_freq[i] - radian_frequency / time_step))
        for i in range(data_length)
    ]

    # get the transform at the inertial and "anti-inertial" frequencies
    # extract the values of wp and wn at the calculated index as a function of time
    # positive is anticyclonic (inertial) in the southern hemisphere
    # negative is anticyclonic (inertial) in the northern hemisphere
    wp = wp[frequency_bins, np.arange(0, data_length)]
    wn = wn[frequency_bins, np.arange(0, data_length)]

    # indices of northern latitude points
    north = latitude >= 0

    # initialize the zonal and meridional components of inertial displacements
    wxhat = np.zeros_like(latitude, dtype=np.complex64)
    wyhat = np.zeros_like(latitude, dtype=np.complex64)
    # equations are x+ = 0.5*(z+ + z-) and y+ = -0.5*1j*(z+ - z-)
    if any(north):
        wxhat[north] = wn[north] / np.sqrt(2)
        wyhat[north] = 1j * wn[north] / np.sqrt(2)
    if any(~north):
        wxhat[~north] = wp[~north] / np.sqrt(2)
        wyhat[~north] = -1j * wp[~north] / np.sqrt(2)

    # inertial displacement in meters
    xhat = np.real(wxhat)
    yhat = np.real(wyhat)

    return xhat, yhat


def residual_position_from_displacement(
    longitude: Union[float, np.ndarray, xr.DataArray],
    latitude: Union[float, np.ndarray, xr.DataArray],
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
) -> Union[Tuple[float], Tuple[np.ndarray]]:
    """
    Return residual longitudes and latitudes along a trajectory on the spherical Earth
    after correcting for zonal and meridional displacements x and y in meters.

    This is applicable as an example when one seeks to correct a trajectory for
    horizontal oscillations due to inertial motions, tides, etc.

    Parameters
    ----------
    longitude : float or array-like
        Longitude in degrees.
    latitude : float or array-like
        Latitude in degrees.
    x : float or np.ndarray
        Zonal displacement in meters.
    y : float or np.ndarray
        Meridional displacement in meters.

    Returns
    -------
    residual_longitude : float or np.ndarray
        Residual longitude after correcting for zonal displacement, in degrees.
    residual_latitude : float or np.ndarray
        Residual latitude after correcting for meridional displacement, in degrees.

    Examples
    --------
    Obtain the new geographical position for a displacement of 1/360-th of the
    circumference of the Earth from original position (longitude,latitude) = (1,0):

    >>> from clouddrift.sphere import EARTH_RADIUS_METERS
    >>> residual_position_from_displacement(1,0,2 * np.pi * EARTH_RADIUS_METERS / 360,0)
    (0.0, 0.0)
    """
    # convert to numpy arrays to insure consistent outputs
    if isinstance(longitude, xr.DataArray):
        longitude = longitude.to_numpy()
    if isinstance(latitude, xr.DataArray):
        latitude = latitude.to_numpy()

    latitudehat = 180 / np.pi * y / EARTH_RADIUS_METERS
    longitudehat = (
        180 / np.pi * x / (EARTH_RADIUS_METERS * np.cos(np.radians(latitude)))
    )

    residual_latitude = latitude - latitudehat
    residual_longitude = recast_lon360(
        np.degrees(np.angle(np.exp(1j * np.radians(longitude - longitudehat))))
    )

    return residual_longitude, residual_latitude


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
    if not u.shape == v.shape:
        raise ValueError("u and v must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(u.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(u.shape) - 1}])."
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
            x[..., n + 1], y[..., n + 1] = position_from_distance_and_bearing(
                x[..., n], y[..., n], distances[..., n], bearings[..., n]
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

    Examples
    --------
    Simple integration on a sphere, using the forward scheme by default:

    >>> import numpy as np
    >>> from clouddrift.kinematics import velocity_from_position
    >>> lon = np.array([0., 1., 3., 6.])
    >>> lat = np.array([0., 1., 2., 3.])
    >>> time = np.array([0., 1., 2., 3.]) * 1e5
    >>> u, v = velocity_from_position(lon, lat, time)
    >>> u
    array([1.11307541, 2.22513331, 3.33515501, 3.33515501])
    >>> v
    array([1.11324496, 1.11409224, 1.1167442 , 1.1167442 ])

    Integration on a Cartesian plane, using the forward scheme by default:

    >>> x = np.array([0., 1., 3., 6.])
    >>> y = np.array([0., 1., 2., 3.])
    >>> time = np.array([0., 1., 2., 3.])
    >>> u, v = velocity_from_position(x, y, time, coord_system="cartesian")
    >>> u
    array([1., 2., 3., 3.])
    >>> v
    array([1., 1., 1., 1.])

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
            distances = distance(x_[..., :-1], y_[..., :-1], x_[..., 1:], y_[..., 1:])
            bearings = bearing(x_[..., :-1], y_[..., :-1], x_[..., 1:], y_[..., 1:])
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
            distances = distance(x_[..., :-1], y_[..., :-1], x_[..., 1:], y_[..., 1:])
            bearings = bearing(x_[..., :-1], y_[..., :-1], x_[..., 1:], y_[..., 1:])
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
            distances = distance(x1, y1, x2, y2)
            bearings = bearing(x1, y1, x2, y2)
            dx[..., 1:-1] = distances * np.cos(bearings)
            dy[..., 1:-1] = distances * np.sin(bearings)

            # Boundary values
            distance1 = distance(x_[..., 0], y_[..., 0], x_[..., 1], y_[..., 1])
            bearing1 = bearing(x_[..., 0], y_[..., 0], x_[..., 1], y_[..., 1])
            dx[..., 0] = distance1 * np.cos(bearing1)
            dy[..., 0] = distance1 * np.sin(bearing1)
            distance2 = distance(x_[..., -2], y_[..., -2], x_[..., -1], y_[..., -1])
            bearing2 = bearing(x_[..., -2], y_[..., -2], x_[..., -1], y_[..., -1])
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


def spin(
    u: np.ndarray,
    v: np.ndarray,
    time: np.ndarray,
    difference_scheme: Optional[str] = "forward",
    time_axis: Optional[int] = -1,
) -> Union[float, np.ndarray]:
    """Compute spin continuously from velocities and times.

    Spin is traditionally (Sawford, 1999; Veneziani et al., 2005) defined as
    (<u'dv' - v'du'>) / (2 dt EKE) where u' and v' are eddy-perturbations of the
    velocity field, EKE is eddy kinetic energy, dt is the time step, and du' and
    dv' are velocity component increments during dt, and < > denotes ensemble
    average.

    To allow computing spin based on full velocity fields, this function does
    not do any demeaning of the velocity fields. If you need the spin based on
    velocity anomalies, ensure to demean the velocity fields before passing
    them to this function. This function also returns instantaneous spin values,
    so the rank of the result is not reduced relative to the input.

    ``u``, ``v``, and ``time`` can be multi-dimensional arrays. If the time
    axis, along which the finite differencing is performed, is not the last one
    (i.e. ``u.shape[-1]``), use the time_axis optional argument to specify along
    which the spin should be calculated. u, v, and time must either have the
    same shape, or time must be a 1-d array with the same length as
    ``u.shape[time_axis]``.

    Difference scheme can be one of three values:

        1. "forward" (default): finite difference is evaluated as ``dx[i] = dx[i+1] - dx[i]``;
        2. "backward": finite difference is evaluated as ``dx[i] = dx[i] - dx[i-1]``;
        3. "centered": finite difference is evaluated as ``dx[i] = (dx[i+1] - dx[i-1]) / 2``.

    Forward and backward schemes are effectively the same except that the
    position at which the velocity is evaluated is shifted one element down in
    the backward scheme relative to the forward scheme. In the case of a
    forward or backward difference scheme, the last or first element of the
    velocity, respectively, is extrapolated from its neighboring point. In the
    case of a centered difference scheme, the start and end boundary points are
    evaluated using the forward and backward difference scheme, respectively.

    Parameters
    ----------
    u : np.ndarray
        Zonal velocity
    v : np.ndarray
        Meridional velocity
    time : array-like
        Time
    difference_scheme : str, optional
        Difference scheme to use; possible values are "forward", "backward", and "centered".
    time_axis : int, optional
        Axis along which the time varies (default is -1)

    Returns
    -------
    s : float or np.ndarray
        Spin

    Raises
    ------
    ValueError
        If u and v do not have the same shape.
        If the time axis is outside of the valid range ([-1, N-1]).
        If lengths of u, v, and time along time_axis are not equal.
        If difference_scheme is not "forward", "backward", or "centered".

    Examples
    --------
    >>> from clouddrift.kinematics import spin
    >>> import numpy as np
    >>> u = np.array([1., 2., -1., 4.])
    >>> v = np.array([1., 3., -2., 1.])
    >>> time = np.array([0., 1., 2., 3.])
    >>> spin(u, v, time)
    array([ 0.5       , -0.07692308,  1.4       ,  0.41176471])

    Use ``difference_scheme`` to specify an alternative finite difference
    scheme for the velocity differences:

    >>> spin(u, v, time, difference_scheme="centered")
    array([0.5       , 0.        , 0.6       , 0.41176471])
    >>> spin(u, v, time, difference_scheme="backward")
    array([ 0.5       ,  0.07692308, -0.2       ,  0.41176471])

    References
    ----------
    * Sawford, B.L., 1999. Rotation of trajectories in Lagrangian stochastic models of turbulent dispersion. Boundary-layer meteorology, 93, pp.411-424. https://doi.org/10.1023/A:1002114132715
    * Veneziani, M., Griffa, A., Garraffo, Z.D. and Chassignet, E.P., 2005. Lagrangian spin parameter and coherent structures from trajectories released in a high-resolution ocean model. Journal of Marine Research, 63(4), pp.753-788. https://elischolar.library.yale.edu/journal_of_marine_research/100/
    """
    if not u.shape == v.shape:
        raise ValueError("u and v arrays must have the same shape.")

    if not time.shape == u.shape:
        if not time.size == u.shape[time_axis]:
            raise ValueError("time must have the same length as u along time_axis.")

    # axis must be in valid range
    if time_axis < -1 or time_axis > len(u.shape) - 1:
        raise ValueError(
            f"axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(u.shape) - 1}])."
        )

    # Swap axes so that we can differentiate along the last axis.
    # This is a syntax convenience rather than memory access optimization:
    # np.swapaxes returns a view of the array, not a copy, if the input is a
    # NumPy array. Otherwise, it returns a copy.
    u = np.swapaxes(u, time_axis, -1)
    v = np.swapaxes(v, time_axis, -1)
    time = np.swapaxes(time, time_axis, -1)

    if not time.shape == u.shape:
        # time is 1-d array; broadcast to u.shape.
        time = np.broadcast_to(time, u.shape)

    du = np.empty(u.shape)
    dv = np.empty(v.shape)
    dt = np.empty(time.shape)

    if difference_scheme == "forward":
        du[..., :-1] = np.diff(u)
        du[..., -1] = du[..., -2]
        dv[..., :-1] = np.diff(v)
        dv[..., -1] = dv[..., -2]
        dt[..., :-1] = np.diff(time)
        dt[..., -1] = dt[..., -2]
    elif difference_scheme == "backward":
        du[..., 1:] = np.diff(u)
        du[..., 0] = du[..., 1]
        dv[..., 1:] = np.diff(v)
        dv[..., 0] = dv[..., 1]
        dt[..., 1:] = np.diff(time)
        dt[..., 0] = dt[..., 1]
    elif difference_scheme == "centered":
        du[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / 2
        du[..., 0] = u[..., 1] - u[..., 0]
        du[..., -1] = u[..., -1] - u[..., -2]
        dv[..., 1:-1] = (v[..., 2:] - v[..., :-2]) / 2
        dv[..., 0] = v[..., 1] - v[..., 0]
        dv[..., -1] = v[..., -1] - v[..., -2]
        dt[..., 1:-1] = (time[..., 2:] - time[..., :-2]) / 2
        dt[..., 0] = time[..., 1] - time[..., 0]
        dt[..., -1] = time[..., -1] - time[..., -2]
    else:
        raise ValueError(
            'difference_scheme must be "forward", "backward", or "centered".'
        )

    # Compute spin
    s = (u * dv - v * du) / (2 * dt * kinetic_energy(u, v))

    return np.swapaxes(s, time_axis, -1)

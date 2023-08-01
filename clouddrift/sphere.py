"""
This module provides functions for calculations on the surface of a sphere,
such as functions to recast longitude values to a desired range, as well as
functions to project from tangent plane to sphere and vice versa.
"""

import numpy as np
from typing import Optional, Tuple
import xarray as xr

EARTH_RADIUS_METERS = 6.3781e6


def distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Return elementwise great circle distance in meters between one or more
    points from arrays of their latitudes and longitudes, using the Haversine
    formula.

    d = 2⋅r⋅asin √[sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)]

    where (φ, λ) is (lat, lon) in radians and r is the radius of the sphere in
    meters.

    Parameters
    ----------
    lat1 : np.ndarray
        Latitudes of the first set of points, in degrees
    lon1 : np.ndarray
        Longitudes of the first set of points, in degrees
    lat2 : np.ndarray
        Latitudes of the second set of points, in degrees
    lon2 : np.ndarray
        Longitudes of the second set of points, in degrees

    Returns
    -------
    out : np.ndarray
        Great circle distance
    """

    # Input coordinates are in degrees; convert to radians.
    # If any of the input arrays are xr.DataArray, extract the values first
    # because Xarray enforces alignment between coordinates.
    if type(lat1) is xr.DataArray:
        lat1_rad = np.deg2rad(lat1.values)
    else:
        lat1_rad = np.deg2rad(lat1)
    if type(lon1) is xr.DataArray:
        lon1_rad = np.deg2rad(lon1.values)
    else:
        lon1_rad = np.deg2rad(lon1)
    if type(lat2) is xr.DataArray:
        lat2_rad = np.deg2rad(lat2.values)
    else:
        lat2_rad = np.deg2rad(lat2)
    if type(lon2) is xr.DataArray:
        lon2_rad = np.deg2rad(lon2.values)
    else:
        lon2_rad = np.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    h = (
        np.sin(0.5 * dlat) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(0.5 * dlon) ** 2
    )

    return 2 * np.arcsin(np.sqrt(h)) * EARTH_RADIUS_METERS


def bearing(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Return elementwise initial (forward) bearing in radians from arrays of
    latitude and longitude in degrees, based on the spherical law of cosines.

    The formula is:

    θ = atan2(cos φ1 ⋅ sin φ2 - sin φ1 ⋅ cos φ2 ⋅ cos Δλ, sin Δλ ⋅ cos φ2)

    where (φ, λ) is (lat, lon) and θ is bearing, all in radians.
    Bearing is defined as zero toward East and positive counterclockwise.

    Parameters
    ----------
    lat1 : np.ndarray
        Latitudes of the first set of points, in degrees
    lon1 : np.ndarray
        Longitudes of the first set of points, in degrees
    lat2 : np.ndarray
        Latitudes of the second set of points, in degrees
    lon2 : np.ndarray
        Longitudes of the second set of points, in degrees

    Returns
    -------
    theta : np.ndarray
        Bearing angles in radians
    """

    # Input coordinates are in degrees; convert to radians.
    # If any of the input arrays are xr.DataArray, extract the values first
    # because Xarray enforces alignment between coordinates.
    if type(lat1) is xr.DataArray:
        lat1_rad = np.deg2rad(lat1.values)
    else:
        lat1_rad = np.deg2rad(lat1)
    if type(lon1) is xr.DataArray:
        lon1_rad = np.deg2rad(lon1.values)
    else:
        lon1_rad = np.deg2rad(lon1)
    if type(lat2) is xr.DataArray:
        lat2_rad = np.deg2rad(lat2.values)
    else:
        lat2_rad = np.deg2rad(lat2)
    if type(lon2) is xr.DataArray:
        lon2_rad = np.deg2rad(lon2.values)
    else:
        lon2_rad = np.deg2rad(lon2)

    dlon = lon2_rad - lon1_rad

    theta = np.arctan2(
        np.cos(lat1_rad) * np.sin(lat2_rad)
        - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon),
        np.sin(dlon) * np.cos(lat2_rad),
    )

    return theta


def position_from_distance_and_bearing(
    lat: float, lon: float, distance: float, bearing: float
) -> Tuple[float, float]:
    """Return elementwise new position in degrees from arrays of latitude and
    longitude in degrees, distance in meters, and bearing in radians, based on
    the spherical law of cosines.

    The formula is:

    φ2 = asin( sin φ1 ⋅ cos δ + cos φ1 ⋅ sin δ ⋅ cos θ )
    λ2 = λ1 + atan2( sin θ ⋅ sin δ ⋅ cos φ1, cos δ − sin φ1 ⋅ sin φ2 )

    where (φ, λ) is (lat, lon) and θ is bearing, all in radians.
    Bearing is defined as zero toward East and positive counterclockwise.

    Parameters
    ----------
    lat : float
        Latitude of the first set of points, in degrees
    lon : float
        Longitude of the first set of points, in degrees
    distance : array_like
        Distance in meters
    bearing : array_like
        Bearing angles in radians

    Returns
    -------
    lat2 : array_like
        Latitudes of the second set of points, in degrees, in the range [-90, 90]
    lon2 : array_like
        Longitudes of the second set of points, in degrees, in the range [-180, 180]
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    distance_rad = distance / EARTH_RADIUS_METERS

    lat2_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(distance_rad)
        + np.cos(lat_rad) * np.sin(distance_rad) * np.sin(bearing)
    )
    lon2_rad = lon_rad + np.arctan2(
        np.cos(bearing) * np.sin(distance_rad) * np.cos(lat_rad),
        np.cos(distance_rad) - np.sin(lat_rad) * np.sin(lat2_rad),
    )

    return np.rad2deg(lat2_rad), np.rad2deg(lon2_rad)


def recast_lon(lon: np.ndarray, lon0: Optional[float] = -180) -> np.ndarray:
    """Recast (convert) longitude values to a selected range of 360 degrees starting from ``lon0``.

    Parameters
    ----------
    lon : np.ndarray or float
        An N-d array of longitudes in degrees
    lon0 : float, optional
        Starting longitude of the recasted range (default -180).

    Returns
    -------
    np.ndarray or float
        Converted longitudes in the range `[lon0, lon0+360]`

    Examples
    --------

    By default, ``recast_lon`` converts longitude values to the range
    `[-180, 180]`:

    >>> recast_lon(200)
    -160

    The range of the output longitude is controlled by ``lon0``.
    For example, with ``lon0 = 0``, the longitude values are converted to the
    range `[0, 360]`.

    >>> recast_lon(200, -180)
    -160

    With ``lon0 = 20``, longitude values are converted to range `[20, 380]`,
    which can be useful to avoid cutting the major ocean basins.

    >>> recast_lon(10, 20)
    370

    See Also
    --------
    :func:`recast_lon360`, :func:`recast_lon180`
    """
    return np.mod(lon - lon0, 360) + lon0


def recast_lon360(lon: np.ndarray) -> np.ndarray:
    """Recast (convert) longitude values to the range `[0, 360]`.
    This is a convenience wrapper around :func:`recast_lon` with ``lon0 = 0``.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of longitudes in degrees

    Returns
    -------
    np.ndarray
        Converted longitudes in the range `[0, 360]`

    Examples
    --------
    >>> recast_lon360(200)
    200

    >>> recast_lon360(-200)
    160

    See Also
    --------
    :func:`recast_lon`, :func:`recast_lon180`
    """
    return recast_lon(lon, 0)


def recast_lon180(lon: np.ndarray) -> np.ndarray:
    """Recast (convert) longitude values to the range `[-180, 180]`.
    This is a convenience wrapper around :func:`recast_lon` with ``lon0 = -180``.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of longitudes in degrees

    Returns
    -------
    np.ndarray
        Converted longitudes in the range `[-180, 180]`

    Examples
    --------
    >>> recast_lon180(200)
    -160

    >>> recast_lon180(-200)
    160

    See Also
    --------
    :func:`recast_lon`, :func:`recast_lon360`
    """
    return recast_lon(lon, -180)


def plane_to_sphere(
    x: np.ndarray, y: np.ndarray, lon_origin: float = 0, lat_origin: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates on a plane to spherical coordinates.

    The arrays of input zonal and meridional displacements ``x`` and ``y`` are
    assumed to follow a contiguous trajectory. The spherical coordinate of each
    successive point is determined by following a great circle path from the
    previous point. The spherical coordinate of the first point is determined by
    following a great circle path from the origin, by default (0, 0).

    The output arrays have the same floating-point output type as the input.

    If projecting multiple trajectories onto the same plane, use
    :func:`apply_ragged` for highest accuracy.

    Parameters
    ----------
    x : np.ndarray
        An N-d array of zonal displacements in meters
    y : np.ndarray
        An N-d array of meridional displacements in meters
    lon_origin : float, optional
        Origin longitude of the tangent plane in degrees, default 0
    lat_origin : float, optional
        Origin latitude of the tangent plane in degrees, default 0

    Returns
    -------
    lon : np.ndarray
        Longitude in degrees
    lat : np.ndarray
        Latitude in degrees

    Examples
    --------
    >>> plane_to_sphere(np.array([0., 0.]), np.array([0., 1000.]))
    (array([0.00000000e+00, 5.50062664e-19]), array([0.       , 0.0089832]))

    You can also specify an origin longitude and latitude:

    >>> plane_to_sphere(np.array([0., 0.]), np.array([0., 1000.]), lon_origin=1, lat_origin=0)
    (array([1., 1.]), array([0.       , 0.0089832]))

    Raises
    ------
    AttributeError
        If ``x`` and ``y`` are not NumPy arrays

    See Also
    --------
    :func:`sphere_to_plane`
    """
    lon = np.empty_like(x)
    lat = np.empty_like(y)

    # Cartesian distances between each point
    dx = np.diff(x, prepend=0)
    dy = np.diff(y, prepend=0)

    distances = np.sqrt(dx**2 + dy**2)
    bearings = np.arctan2(dy, dx)

    # Compute spherical coordinates following great circles between each
    # successive point.
    lat[..., 0], lon[..., 0] = position_from_distance_and_bearing(
        lat_origin, lon_origin, distances[..., 0], bearings[..., 0]
    )
    for n in range(1, lon.shape[-1]):
        lat[..., n], lon[..., n] = position_from_distance_and_bearing(
            lat[..., n - 1], lon[..., n - 1], distances[..., n], bearings[..., n]
        )

    return lon, lat


def sphere_to_plane(
    lon: np.ndarray, lat: np.ndarray, lon_origin: float = 0, lat_origin: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert spherical coordinates to a tangent (Cartesian) plane.

    The arrays of input longitudes and latitudes are assumed to be following
    a contiguous trajectory. The Cartesian coordinate of each successive point
    is determined by following a great circle path from the previous point.
    The Cartesian coordinate of the first point is determined by following a
    great circle path from the origin, by default (0, 0).

    The output arrays have the same floating-point output type as the input.

    If projecting multiple trajectories onto the same plane, use
    :func:`apply_ragged` for highest accuracy.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of longitudes in degrees
    lat : np.ndarray
        An N-d array of latitudes in degrees
    lon_origin : float, optional
        Origin longitude of the tangent plane in degrees, default 0
    lat_origin : float, optional
        Origin latitude of the tangent plane in degrees, default 0

    Returns
    -------
    x : np.ndarray
        x-coordinates on the tangent plane
    y : np.ndarray
        y-coordinates on the tangent plane

    Examples
    --------
    >>> sphere_to_plane(np.array([0., 1.]), np.array([0., 0.]))
    (array([     0.        , 111318.84502145]), array([0., 0.]))

    You can also specify an origin longitude and latitude:

    >>> sphere_to_plane(np.array([0., 1.]), np.array([0., 0.]), lon_origin=1, lat_origin=0)
    (array([-111318.84502145,       0.        ]),
     array([1.36326267e-11, 1.36326267e-11]))

    Raises
    ------
    AttributeError
        If ``lon`` and ``lat`` are not NumPy arrays

    See Also
    --------
    :func:`plane_to_sphere`
    """
    x = np.empty_like(lon)
    y = np.empty_like(lat)

    distances = np.empty_like(x)
    bearings = np.empty_like(x)

    # Distance and bearing of the starting point relative to the origin
    distances[0] = distance(lat_origin, lon_origin, lat[..., 0], lon[..., 0])
    bearings[0] = bearing(lat_origin, lon_origin, lat[..., 0], lon[..., 0])

    # Distance and bearing of the remaining points
    distances[1:] = distance(lat[..., :-1], lon[..., :-1], lat[..., 1:], lon[..., 1:])
    bearings[1:] = bearing(lat[..., :-1], lon[..., :-1], lat[..., 1:], lon[..., 1:])

    dx = distances * np.cos(bearings)
    dy = distances * np.sin(bearings)

    x[..., :] = np.cumsum(dx, axis=-1)
    y[..., :] = np.cumsum(dy, axis=-1)

    return x, y

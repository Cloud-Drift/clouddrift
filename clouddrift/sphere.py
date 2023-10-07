"""
This module provides functions for spherical geometry calculations.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import warnings

EARTH_RADIUS_METERS = 6.3781e6
EARTH_DAY_SECONDS = 86164.091
EARTH_ROTATION_RATE = 2 * np.pi / EARTH_DAY_SECONDS


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
        Converted longitudes in the range `[lon0, lon0+360[`

    Examples
    --------

    By default, ``recast_lon`` converts longitude values to the range
    `[-180, 180[`:

    >>> recast_lon(200)
    -160

    >>> recast_lon(180)
    -180

    The range of the output longitude is controlled by ``lon0``.
    For example, with ``lon0 = 0``, the longitude values are converted to the
    range `[0, 360[`.

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
    """Recast (convert) longitude values to the range `[0, 360[`.
    This is a convenience wrapper around :func:`recast_lon` with ``lon0 = 0``.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of longitudes in degrees

    Returns
    -------
    np.ndarray
        Converted longitudes in the range `[0, 360[`

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
    """Recast (convert) longitude values to the range `[-180, 180[`.
    This is a convenience wrapper around :func:`recast_lon` with ``lon0 = -180``.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of longitudes in degrees

    Returns
    -------
    np.ndarray
        Converted longitudes in the range `[-180, 180[`

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


def spherical_to_cartesian(
    lon: np.ndarray,
    lat: np.ndarray,
    radius: Optional[float] = EARTH_RADIUS_METERS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts latitude and longitude on a spherical body to
     three-dimensional Cartesian coordinates.

    The Cartesian coordinate system is a right-handed system whose
    origin lies at the center of a sphere.  It is oriented with the
    Z-axis passing through the poles and the X-axis passing through
    the point lon = 0, lat = 0. This function is inverted by `cartesian_to_spherical`.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of longitudes in degrees.
    lat : np.ndarray
        An N-d array of latitudes in degrees.
    radius: float, optional
        The radius of the spherical body in meters. The default assumes the Earth with
        EARTH_RADIUS_METERS = 6.3781e6.

    Returns
    -------
    x : np.ndarray
        x-coordinates in 3D in meters.
    y : np.ndarray
        y-coordinates in 3D in meters.
    z : np.ndarray
        z-coordinates in 3D in meters.

    Examples
    --------
    >>> spherical_to_cartesian(np.array([0, 45]),np.array([0, 45]))
    (array([6378100., 3189050.]),
    array([      0., 3189050.]),
    array([      0.        , 4509997.76108592]))

    >>> spherical_to_cartesian(np.array([0, 45, 90]),np.array([0, 90, 180]),radius=1)
    (array([ 1.00000000e+00,  4.32978028e-17, -6.12323400e-17]),
    array([ 0.00000000e+00,  4.32978028e-17, -1.00000000e+00]),
    array([0.0000000e+00, 1.0000000e+00, 1.2246468e-16]))

    >>> x, y, z = spherical_to_cartesian(np.array([0,5]),np.array([0,5]))

    Raises
    ------
    AttributeError
        If ``lon`` and ``lat`` are not NumPy arrays.

    See Also
    --------
    :func:`cartesian_to_spherical`
    """
    lonr, latr = np.deg2rad(lon), np.deg2rad(lat)

    x = radius * np.cos(latr) * np.cos(lonr)
    y = radius * np.cos(latr) * np.sin(lonr)
    z = radius * np.sin(latr)

    return x, y, z


def cartesian_to_spherical(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts Cartesian three-dimensional coordinates to latitude and longitude on a
    spherical body.

    The Cartesian coordinate system is a right-handed system whose
    origin lies at the center of the sphere.  It is oriented with the
    Z-axis passing through the poles and the X-axis passing through
    the point lon = 0, lat = 0. This function is inverted by `spherical_to_cartesian`.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates in 3D.
    y : np.ndarray
        y-coordinates in 3D.
    z : np.ndarray
        z-coordinates in 3D.

    Returns
    -------
    lon : np.ndarray
        An N-d array of longitudes in degrees in range [-180, 180].
    lat : np.ndarray
        An N-d array of latitudes in degrees.

    Examples
    --------
    >>> x = EARTH_RADIUS_METERS * np.cos(np.deg2rad(45))
    >>> y = EARTH_RADIUS_METERS * np.cos(np.deg2rad(45))
    >>> z = 0 * x
    >>> cartesian_to_spherical(x, y, z)
    (44.99999999999985, 0.0)

    `cartesian_to_spherical` is inverted by `spherical_to_cartesian`:

    >>> x, y, z = spherical_to_cartesian(np.array([45]),np.array(0))
    >>> cartesian_to_spherical(x, y, z)
    (array([45.]), array([0.]))

    Raises
    ------
    AttributeError
        If ``x``, ``y``, and ``z`` are not NumPy arrays.

    See Also
    --------
    :func:`spherical_to_cartesian`
    """

    R = np.sqrt(x**2 + y**2 + z**2)
    x /= R
    y /= R
    z /= R

    lon = recast_lon180(np.rad2deg(np.imag(np.log((x + 1j * y)))))
    lat = np.rad2deg(np.arcsin(z))

    return lon, lat


def cartesian_to_tangentplane(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    w: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
    latitude: Union[float, np.ndarray],
) -> Union[Tuple[float], Tuple[np.ndarray]]:
    """
    Project a three-dimensional Cartesian vector on a plane tangent to
    a spherical Earth.

    The Cartesian coordinate system is a right-handed system whose
    origin lies at the center of a sphere.  It is oriented with the
    Z-axis passing through the north pole at lat = 90, the X-axis passing through
    the point lon = 0, lat = 0, and the Y-axis passing through the point lon = 90,
    lat = 0.

    Parameters
    ----------
        u : float or np.ndarray
            First component of Cartesian vector.
        v : float or np.ndarray
            Second component of Cartesian vector.
        w : float or np.ndarray
            Third component of Cartesian vector.
        longitude : float or np.ndarray
            Longitude in degrees of tangent point of plane.
        latitude : float or np.ndarray
            Latitude in degrees of tangent point of plane.

    Returns
    -------
        up: float or np.ndarray
            First component of projected vector on tangent plane (positive eastward).
        vp: float or np.ndarray
            Second component of projected vector on tangent plane (positive northward).

    Raises
    ------
    Warning
        Raised if the input latitude is not in the expected range [-90, 90].

    Examples
    --------
    >>> u, v = cartesian_to_tangentplane(1, 1, 1, 45, 90)

    See Also
    --------
    :func:`tangentplane_to_cartesian`
    """
    if np.any(latitude < -90) or np.any(latitude > 90):
        warnings.warn("Input latitude outside of range [-90,90].")

    phi = np.radians(latitude)
    theta = np.radians(longitude)
    u_projected = v * np.cos(theta) - u * np.sin(theta)
    v_projected = (
        w * np.cos(phi)
        - u * np.cos(theta) * np.sin(phi)
        - v * np.sin(theta) * np.sin(phi)
    )
    # JML says vh = w.*cos(phi)-u.*cos(theta).*sin(phi)-v.*sin(theta).*sin(phi) but vh=w./cos(phi) is the same
    return u_projected, v_projected


def tangentplane_to_cartesian(
    up: Union[float, np.ndarray],
    vp: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
    latitude: Union[float, np.ndarray],
) -> Union[Tuple[float], Tuple[np.ndarray]]:
    """
    Return the three-dimensional Cartesian components of a vector contained in
    a plane tangent to a spherical Earth.

    The Cartesian coordinate system is a right-handed system whose
    origin lies at the center of a sphere.  It is oriented with the
    Z-axis passing through the north pole at lat = 90, the X-axis passing through
    the point lon = 0, lat = 0, and the Y-axis passing through the point lon = 90,
    lat = 0.

    Parameters
    ----------
        up: float or np.ndarray
            First component of vector on tangent plane (positive eastward).
        vp: float or np.ndarray
            Second component of vector on tangent plane (positive northward).
        longitude : float or np.ndarray
            Longitude in degrees of tangent point of plane.
        latitude : float or np.ndarray
            Latitude in degrees of tangent point of plane.

    Returns
    -------
        u : float or np.ndarray
            First component of Cartesian vector.
        v : float or np.ndarray
            Second component of Cartesian vector.
        w : float or np.ndarray
            Third component of Cartesian vector.

    Examples
    --------
    >>> u, v, w = tangentplane_to_cartesian(1, 1, 45, 90)

    Notes
    -----
    This function is inverted by `cartesian_to_tangetplane`.

    See Also
    --------
    :func:`cartesian_to_tangentplane`
    """
    phi = np.radians(latitude)
    theta = np.radians(longitude)
    u = -up * np.sin(theta) - vp * np.sin(phi) * np.cos(theta)
    v = up * np.cos(theta) - vp * np.sin(phi) * np.sin(theta)
    w = vp * np.cos(phi)

    return u, v, w


def coriolis_frequency(
    latitude: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Return the Coriolis frequency or commonly known `f` parameter in geophysical fluid dynamics.

    Parameters
    ----------
    latitude : float or np.ndarray
        Latitude in degrees.

    Returns
    -------
    f : float or np.ndarray
        Signed Coriolis frequency in radian per seconds.

    Examples
    --------
    >>> f = coriolis_frequency(np.array([0, 45, 90]))

    """
    f = 2 * EARTH_ROTATION_RATE * np.sin(np.radians(latitude))

    return f

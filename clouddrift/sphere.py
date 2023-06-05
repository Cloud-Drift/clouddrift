from clouddrift import haversine
import numpy as np
from typing import Optional, Tuple


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

    This function uses 64-bit floats for all intermediate calculations,
    regardless of the type of input arrays, to avoid loss of precision.
    The output is thus also in 64-bit floats.

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
    Tuple[np.ndarray, np.ndarray]
        Longitude and latitude in degrees

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
    lon = np.empty(x.shape, dtype=np.float64)
    lat = np.empty(y.shape, dtype=np.float64)

    # Cartesian distances between each point
    dx = np.diff(x, prepend=0)
    dy = np.diff(y, prepend=0)

    distances = np.sqrt(dx**2 + dy**2)
    bearings = np.arctan2(dy, dx)

    # Compute spherical coordinates following great circles between each
    # successive point.
    lat[..., 0], lon[..., 0] = haversine.position_from_distance_and_bearing(
        lat_origin, lon_origin, distances[..., 0], bearings[..., 0]
    )
    for n in range(1, lon.shape[-1]):
        lat[..., n], lon[..., n] = haversine.position_from_distance_and_bearing(
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

    This function uses 64-bit floats for all intermediate calculations,
    regardless of the type of input arrays, to avoid loss of precision.
    The output is thus also in 64-bit floats.

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
    Tuple[np.ndarray, np.ndarray]
        x- and y-coordinates of the tangent plane

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
    x = np.empty(lon.shape, dtype=np.float64)
    y = np.empty(lat.shape, dtype=np.float64)
    distances = np.empty(lon.shape, dtype=np.float64)
    bearings = np.empty(lon.shape, dtype=np.float64)

    # Distance and bearing of the starting point relative to the origin
    distances[0] = haversine.distance(lat_origin, lon_origin, lat[..., 0], lon[..., 0])
    bearings[0] = haversine.bearing(lat_origin, lon_origin, lat[..., 0], lon[..., 0])

    # Distance and bearing of the remaining points
    distances[1:] = haversine.distance(
        lat[..., :-1], lon[..., :-1], lat[..., 1:], lon[..., 1:]
    )
    bearings[1:] = haversine.bearing(
        lat[..., :-1], lon[..., :-1], lat[..., 1:], lon[..., 1:]
    )

    dx = distances * np.cos(bearings)
    dy = distances * np.sin(bearings)

    x[..., :] = np.cumsum(dx, axis=-1)
    y[..., :] = np.cumsum(dy, axis=-1)

    return x, y

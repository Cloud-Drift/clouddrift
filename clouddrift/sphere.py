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


def sphere_to_plane(
    lon: np.ndarray, lat: np.ndarray, x_origin: float, y_origin: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert spherical coordinates to a tangent plane.

    Parameters
    ----------
    lat : np.ndarray
        An N-d array of latitudes in degrees
    lon : np.ndarray
        An N-d array of longitudes in degrees
    x_origin : float
        x-coordinate of the origin of the tangent plane
    y_origin : float
        y-coordinate of the origin of the tangent plane

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x- and y-coordinates of the tangent plane
    """
    x = np.empty_like(lon)
    y = np.empty_like(lon)

    distances = haversine.distance(
        lat[..., :-1], lon[..., :-1], lat[..., 1:], lon[..., 1:]
    )
    bearings = haversine.bearing(
        lat[..., :-1], lon[..., :-1], lat[..., 1:], lon[..., 1:]
    )

    dx = distances * np.cos(bearings)
    dy = distances * np.sin(bearings)

    x[..., 0], y[..., 0] = x_origin, y_origin
    x[..., 1:] = np.cumsum(dx, axis=-1)
    y[..., 1:] = np.cumsum(dy, axis=-1)

    return x, y

import numpy as np
from typing import Optional


def recast_lon(lon: np.ndarray, lon0: Optional[float] = -180) -> np.ndarray:
    """Recast (convert) longitude values to a selected range of 360 degrees starting from ``lon0``.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of longitudes in degrees
    lon0 : float, optional
        Starting longitude of the recasted range (default -180).

    Returns
    -------
    np.ndarray
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
    if np.isscalar(lon):
        lon = np.array([lon])

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

import numpy as np
from typing import Optional


def rot(x: np.ndarray) -> np.ndarray:
    """Rotate and return the complex phase of x.

    Parameters
    ----------
    x : np.ndarray
        An N-d array of angles in radians

    Returns
    -------
    np.ndarray
        Rotated and complex phase of x

    Examples
    --------

    .. code-block:: python

        rot(0) # (1+0j)
        rot(np.pi / 2) # approx. (0+1j)
        rot(np.pi) # approx. (-1+0j)
        rot(3 * np.pi / 2) # approx. (0-1j)
    """
    return np.exp(1j * x)


def recast_lon(lon: np.ndarray, lon0: Optional[float] = -180) -> np.ndarray:
    """Recast (convert) longitude values to a selected range of 360 degrees
    starting from ``lon0``.

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

    .. code-block:: python

        recast_lon(200) # -160

    The range of the output longitude is controlled by ``lon0``.
    For example, with ``lon0 = 0``, the longitude values are converted to the
    range `[0, 360]`.

    .. code-block:: python

        recast_lon(200, -180) # -160

    With ``lon0 = 20``, longitude values are converted to range `[20, 380]`,
    which can be useful to avoid cutting the major ocean basins.

    .. code-block:: python

        recast_lon(10, 20) # 370
    """
    if np.isscalar(lon):
        lon = np.array([lon])

    return (
        np.mod(
            np.divide(360, 2 * np.pi)
            * np.unwrap(np.angle(rot(np.divide(2 * np.pi, 360) * (lon - lon0)))),
            360,
        )
        + lon0
    )


def recast_lon360(lon: np.ndarray) -> np.ndarray:
    """Recast (convert) longitude values to the range `[0, 360]`."""
    return recast_lon(lon, 0)


def recast_lon180(lon: np.ndarray) -> np.ndarray:
    """Recast (convert) longitude values to the range `[-180, 180]`."""
    return recast_lon(lon, -180)

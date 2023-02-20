import numpy as np
from typing import Optional


def rot(x: np.ndarray) -> np.ndarray:
    """Complex-valued rotation"""
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi  # convert to [-pi, pi]
    return np.exp(1j * x)


def recast_longitude(lon: np.ndarray, lon0: Optional[float] = -180) -> np.ndarray:
    """Recast (convert) longitude values to a selected range of 360 degrees starting from lon0.

    Parameters
    ----------
    lon : np.ndarray
        An N-d array of x-positions (longitude in degrees)
    lon0 (Optional[float], optional)
        Starting longitude of the recasted range (Default: -180).

    Returns
    -------
    np.ndarray
        Converted longitudes in range [lon0, lon0+360]

    Examples
    --------
    The range of the output longitude is controlled by lon0. For example, with `lon0 = -180`, the longitude values are converted to range `[-180, 180]`.

    .. code-block:: python

        recast_longitude(200, -180)
        > -160

    With `lon0 = 0`, the longitude values are converted to range `[0, 360]`.

    .. code-block:: python

        recast_longitude(200, -180)
        > -160

    With `lon0 = 20`, longitude values are converted to range `[20, 380]`, which can be useful to avoid cutting the major ocean basins.

    .. code-block:: python

        recast_longitude(10, 20)
        > 370

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

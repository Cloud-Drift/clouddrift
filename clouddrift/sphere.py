import numpy as np
from typing import Optional


def rot(x: np.ndarray):
    """Complex-valued rotation"""
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi  # convert to [-pi, pi]
    return np.exp(1j * x)


def recast_longitude(lon: np.ndarray, lon0: Optional[float] = -180):
    """
    Recast longitude in a selected range of 360˚ starting from lon0.

    More precisely, setting:
    - lon0 = -180, longitude range is converted to [-180, 180]
    - lon = 0, longitude range is converted tos [0, 360]
    - lon = 20, longitude range is converted to [20, 380]

    Args:
        lon (array_like): An N-d array of x-positions (longitude in degrees)
        lon0 (float): Starting longitude of the recasted range (Default: -180)
    Returns:
        out (array_like): Recast longitude in range (lon0, lon0+360)

    """
    return (
        np.mod(
            np.divide(360, 2 * np.pi)
            * np.unwrap(np.angle(rot(np.divide(2 * np.pi, 360) * (lon - lon0)))),
            360,
        )
        + lon0
    )

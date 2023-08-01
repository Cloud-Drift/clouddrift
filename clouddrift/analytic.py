"""
This module provides signal processing functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import pandas as pd
import warnings


def anatrans(
    x: Union[list, np.ndarray, xr.DataArray, pd.Series],
    boundary: Optional[str] = "mirror",
) -> np.ndarray:
    """Return the analytic part of a real-valued signal or of a complex-valued
    signal. To obtain the anti-analytic part of a complex-valued signal apply anatrans
    to the conjugate of the input. Anatrans removes the mean of the input signals.

    Parameters
    ----------
    x : np.ndarray
    boundary : str, optional ["mirror", "zeros", "periodic"]

    Returns
    -------
    z : np.ndarray

    Examples
    --------

    To obtain the analytic part of a real-valued signal
    >>> x = np.random.rand(99)
    >>> z = anatrans(x)

    To obtain the analytic and anti-analytic parts of a complex-valued signal
    >>> z = np.random.rand(99)+1j*np.random.rand(99)
    >>> zp = anatrans(z)
    >>> zn = anatrans(np.conj(z))

    Raises
    ------
    ValueError
        If ``boundary not in ["mirror", "zeros", "periodic"]``.
    """
    # assume unidimensional input; add dimension option
    m0 = len(x)

    # remove mean
    x = x - np.mean(x)

    # apply boundary conditions
    if boundary == "mirror":
        x = np.concatenate((np.flip(x), x, np.flip(x)))
    elif boundary == "zeros":
        x = np.concatenate((np.zeros_like(x), x, np.zeros_like(x)))
    elif boundary == "periodic":
        x = np.concatenate((x, x, x))
    else:
        raise ValueError("boundary must be one of 'mirror', 'align', or 'zeros'.")

    m = len(x)

    if np.isrealobj(x):
        z = 2 * np.fft.fft(x)
    else:
        z = np.fft.fft(x)

    if m % 2 == 0:
        z[int(m / 2 + 2) - 1 : int(m + 1) + 1] = 0  # zero negative frequencies
    else:
        z[int((m + 3) / 2) - 1 : int(m + 1) + 1] = 0  # zero negative frequencies

    # inverse Fourier transform
    z = np.fft.ifft(z)

    # return central part
    z = z[int(m0 + 1) - 1 : int(2 * m0 + 1) - 1]

    return z

"""
This module provides signal processing functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import pandas as pd
import warnings


def analytic_transform(
    x: Union[list, np.ndarray, xr.DataArray, pd.Series],
    boundary: Optional[str] = "mirror",
) -> np.ndarray:
    """returns the analytic part of a real-valued signal or of a complex-valued
    signal. To obtain the anti-analytic part of a complex-valued signal apply analytic_transform
    to the conjugate of the input. Analytic_transform removes the mean of the input signals.

    Parameters
    ----------
    x : np.ndarray
        Real- or complex-valued signal
    boundary : str, optional ["mirror", "zeros", "periodic"] optionally specifies the
    boundary condition to be imposed at the edges of the time series. Default is "mirror".

    Returns
    -------
    z : np.ndarray
        Analytic transform of the input signal.

    Examples
    --------

    To obtain the analytic part of a real-valued signal:
    >>> x = np.random.rand(99)
    >>> z = analytic_transform(x)

    To obtain the analytic and anti-analytic parts of a complex-valued signal:
    >>> z = np.random.rand(99)+1j*np.random.rand(99)
    >>> zp = analytic_transform(z)
    >>> zn = analytic_transform(np.conj(z))

    To specify that a periodic boundary condition should be used
    >>> x = np.random.rand(99)
    >>> z = analytic_transform(x,boundary="periodic")

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

    if np.isrealobj(x):
        z = 2 * np.fft.fft(x)
    else:
        z = np.fft.fft(x)

    m = len(x)

    # zero negative frequencies
    if m % 2 == 0:
        z[int(m / 2 + 2) - 1 : int(m + 1) + 1] = 0
    else:
        z[int((m + 3) / 2) - 1 : int(m + 1) + 1] = 0

    # inverse Fourier transform
    z = np.fft.ifft(z)

    # return central part
    z = z[int(m0 + 1) - 1 : int(2 * m0 + 1) - 1]

    return z

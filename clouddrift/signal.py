"""
This module provides signal processing functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import warnings


def analytic_transform(
    x: Union[list, np.ndarray, xr.DataArray],
    boundary: Optional[str] = "mirror",
) -> np.ndarray:
    """Return the analytic part of a real-valued signal or of a complex-valued
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


def rotary_transform(
    u: Union[list, np.ndarray, xr.DataArray],
    v: Union[list, np.ndarray, xr.DataArray],
    boundary: Optional[str] = "mirror",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return time-domain rotary components time series (zp,zn) from Cartesian components time series (u,v).
    Note that zp and zn are both analytic time series which implies that the complex-valued time series
    u+1j*v is recovered by zp+np.conj(zn). The mean of the original complex signal is split evenly between
    the two rotary components.

    If up is the analytic transform of u, and vp the analytic transform of v, then the counterclockwise and
    clockwise components are defined by zp = 0.5*(up+1j*vp), zp = 0.5*(up-1j*vp).
    See as an example Lilly and Olhede (2010), doi: 10.1109/TSP.2009.2031729.

    Parameters
    ----------
    u : np.ndarray
        Real-valued signal, first Cartesian component (zonal, east-west)
    v : np.ndarray
        Real-valued signal, second Cartesian component (meridional, north-south)
    boundary : str, optional ["mirror", "zeros", "periodic"] optionally specifies the
    boundary condition to be imposed at the edges of the time series for the underlying analytic
    transform. Default is "mirror".

    Returns
    -------
    zp : np.ndarray
        Time-domain complex-valued positive (counterclockwise) rotary component.
    zn : np.ndarray
        Time-domain complex-valued negative (clockwise) rotary component.

    Examples
    --------

    To obtain the rotary components of a real-valued signal:
    >>> zp, zn = rotary_transform(u,v)

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.

    See Also
    --------
    :func:`analytic_transform`
    """
    # to implement: input one complex argument instead of two real arguments
    # u and v arrays must have the same shape or list length.
    if type(u) == list and type(v) == list:
        if not len(u) == len(v):
            raise ValueError("u and v must have the same length.")
    else:
        if not u.shape == v.shape:
            raise ValueError("u and v must have the same shape.")

    muv = np.mean(u) + 1j * np.mean(v)

    if muv == xr.DataArray:
        muv = muv.to_numpy()

    up = analytic_transform(u, boundary=boundary)
    vp = analytic_transform(v, boundary=boundary)

    return 0.5 * (up + 1j * vp) + 0.5 * muv, 0.5 * (up - 1j * vp) + 0.5 * np.conj(muv)

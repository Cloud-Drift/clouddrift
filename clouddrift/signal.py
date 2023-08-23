"""
This module provides signal processing functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import warnings


def analytic_transform(
    x: Union[np.ndarray, xr.DataArray],
    boundary: Optional[str] = "mirror",
    time_axis: Optional[int] = -1,
) -> np.ndarray:
    """Return the analytic part of a real-valued signal or of a complex-valued
    signal. To obtain the anti-analytic part of a complex-valued signal apply analytic_transform
    to the conjugate of the input. Analytic_transform removes the mean of the input signals.

    Parameters
    ----------
    x : np.ndarray
        Real- or complex-valued signal
    boundary : str, optional
        The boundary condition to be imposed at the edges of the time series.
        Allowed values are "mirror", "zeros", and "periodic".
        Default is "mirror".
    time_axis : int, optional
        Axis on which the time is defined (default is -1)

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

    To specify that a periodic boundary condition should be used:
    >>> x = np.random.rand(99)
    >>> z = analytic_transform(x, boundary="periodic")

    To specify that the time axis is along the first axis and apply
    zero boundary conditions:
    >>> x = np.random.rand(100, 99)
    >>> z = analytic_transform(x, time_axis=0, boundary="zeros")

    Raises
    ------
    ValueError
        If ``boundary not in ["mirror", "zeros", "periodic"]``.
    """
    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(x.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )

    # Swap the axis to make the time axis last (fast-varying).
    # np.swapaxes returns a view to the input array, so no copy is made.
    if time_axis != -1 and time_axis != len(x.shape) - 1:
        x_ = np.swapaxes(x, time_axis, -1)
    else:
        x_ = x

    # time dimension length
    N = np.shape(x_)[-1]

    # Subtract mean along time axis (-1); convert to np.array for compatibility
    # with xarray.DataArray.
    xa = x_ - np.array(np.mean(x_, axis=-1, keepdims=True))

    # apply boundary conditions
    if boundary == "mirror":
        xa = np.concatenate((np.flip(xa, axis=-1), xa, np.flip(xa, axis=-1)), axis=-1)
    elif boundary == "zeros":
        xa = np.concatenate((np.zeros_like(xa), xa, np.zeros_like(xa)), axis=-1)
    elif boundary == "periodic":
        xa = np.concatenate((xa, xa, xa), axis=-1)
    else:
        raise ValueError("boundary must be one of 'mirror', 'align', or 'zeros'.")

    if np.isrealobj(xa):
        # fft should be taken along last axis
        z = 2 * np.fft.fft(xa)
    else:
        z = np.fft.fft(xa)

    # time dimension of extended time series
    M = np.shape(xa)[-1]

    # zero negative frequencies
    if M % 2 == 0:
        z[..., int(M / 2 + 2) - 1 : int(M + 1) + 1] = 0
        # divide Nyquist component by 2 in even case
        z[..., int(M / 2 + 1) - 1] = z[..., int(M / 2 + 1) - 1] / 2
    else:
        z[..., int((M + 3) / 2) - 1 : int(M + 1) + 1] = 0

    # inverse Fourier transform along last axis
    z = np.fft.ifft(z)

    # return central part
    z = z[..., int(N + 1) - 1 : int(2 * N + 1) - 1]

    # return after reorganizing the axes
    if time_axis != -1 and time_axis != len(x.shape) - 1:
        return np.swapaxes(z, time_axis, -1)
    else:
        return z


def rotary_transform(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    boundary: Optional[str] = "mirror",
    time_axis: Optional[int] = -1,
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
    boundary : str, optional
        The boundary condition to be imposed at the edges of the time series.
        Allowed values are "mirror", "zeros", and "periodic".
        Default is "mirror".
    time_axis : int, optional
        The axis of the time array. Default is -1, which corresponds to the
        last axis.

    Returns
    -------
    zp : np.ndarray
        Time-domain complex-valued positive (counterclockwise) rotary component.
    zn : np.ndarray
        Time-domain complex-valued negative (clockwise) rotary component.

    Examples
    --------

    To obtain the rotary components of a real-valued signal:
    >>> u = np.random.rand(99)
    >>> v = np.random.rand(99)
    >>> zp, zn = rotary_transform(u,v)

    To specify that the time axis is along the first axis, and apply
    zero boundary conditions:
    >>> u = np.random.rand(100,99)
    >>> v = np.random.rand(100,99)
    >>> zp, zn = rotary_transform(u,v,time_axis=0,boundary="zeros")

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.

    See Also
    --------
    :func:`analytic_transform`
    """
    # to implement: input one complex argument instead of two real arguments
    # u and v arrays must have the same shape.
    if not u.shape == v.shape:
        raise ValueError("u and v must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(u.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(u.shape) - 1}])."
        )

    muv = np.mean(u, axis=time_axis, keepdims=True) + 1j * np.mean(
        v, axis=time_axis, keepdims=True
    )

    if type(muv) == xr.DataArray:
        muv = muv.to_numpy()

    up = analytic_transform(u, boundary=boundary, time_axis=time_axis)
    vp = analytic_transform(v, boundary=boundary, time_axis=time_axis)

    zp = 0.5 * (up + 1j * vp) + 0.5 * muv
    zn = 0.5 * (up - 1j * vp) + 0.5 * np.conj(muv)

    return zp, zn

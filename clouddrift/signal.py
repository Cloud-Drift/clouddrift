"""
This module provides signal processing functions.
"""

from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr


def analytic_signal(
    x: Union[np.ndarray, xr.DataArray],
    boundary: Optional[str] = "mirror",
    time_axis: Optional[int] = -1,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return the analytic signal from a real-valued signal or the analytic and
    conjugate analytic signals from a complex-valued signal.

    If the input is a real-valued signal, the analytic signal is calculated as
    the inverse Fourier transform of the positive-frequency part of the Fourier
    transform. If the input is a complex-valued signal, the conjugate analytic signal
    is additionally calculated as the inverse Fourier transform of the positive-frequency
    part of the Fourier transform of the complex conjugate of the input signal.

    For a complex-valued signal, the mean is evenly divided between the analytic and
    conjugate analytic signal.

    The calculation is performed along the last axis of the input array by default.
    Alternatively, the user can specify the time axis of the input. The user can also
    specify the boundary conditions to be applied to the input array (default is "mirror").

    Parameters
    ----------
    x : array_like
        Real- or complex-valued signal.
    boundary : str, optional
        The boundary condition to be imposed at the edges of the time series.
        Allowed values are "mirror", "zeros", and "periodic".
        Default is "mirror".
    time_axis : int, optional
        Axis on which the time is defined (default is -1).

    Returns
    -------
    xa : np.ndarray
        Analytic signal. It is a tuple if the input is a complex-valed signal
        with the first element being the analytic signal and the second element
        being the conjugate analytic signal.

    Examples
    --------

    To obtain the analytic signal of a real-valued signal:

    >>> x = np.random.rand(99)
    >>> xa = analytic_signal(x)

    To obtain the analytic and conjugate analytic signals of a complex-valued signal:

    >>> w = np.random.rand(99)+1j*np.random.rand(99)
    >>> wp, wn = analytic_signal(w)

    To specify that a periodic boundary condition should be used:

    >>> x = np.random.rand(99)
    >>> xa = analytic_signal(x, boundary="periodic")

    To specify that the time axis is along the first axis and apply
    zero boundary conditions:

    >>> x = np.random.rand(100, 99)
    >>> xa = analytic_signal(x, time_axis=0, boundary="zeros")

    Raises
    ------
    ValueError
        If the time axis is outside of the valid range ([-1, N-1]).
        If ``boundary not in ["mirror", "zeros", "periodic"]``.

    References
    ----------
    [1] Gabor D. 1946 Theory of communication. Proc. IEE 93, 429–457. (10.1049/ji-1.1947.0015).

    [2] Lilly JM, Olhede SC. 2010 Bivariate instantaneous frequency and bandwidth.
    IEEE T. Signal Proces. 58, 591–603. (10.1109/TSP.2009.2031729).

    See Also
    --------
    :func:`rotary_to_cartesian`, :func:`cartesian_to_rotary`
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
    mx_ = np.array(np.mean(x_, axis=-1, keepdims=True))
    xa = x_ - mx_

    # apply boundary conditions
    if boundary == "mirror":
        xa = np.concatenate((np.flip(xa, axis=-1), xa, np.flip(xa, axis=-1)), axis=-1)
    elif boundary == "zeros":
        xa = np.concatenate((np.zeros_like(xa), xa, np.zeros_like(xa)), axis=-1)
    elif boundary == "periodic":
        xa = np.concatenate((xa, xa, xa), axis=-1)
    else:
        raise ValueError("boundary must be one of 'mirror', 'align', or 'zeros'.")

    # analytic signal
    xap = np.fft.fft(xa)
    # conjugate analytic signal
    xan = np.fft.fft(np.conj(xa))

    # time dimension of extended time series
    M = np.shape(xa)[-1]

    # zero negative frequencies
    if M % 2 == 0:
        xap[..., int(M / 2 + 2) - 1 : int(M + 1) + 1] = 0
        xan[..., int(M / 2 + 2) - 1 : int(M + 1) + 1] = 0
        # divide Nyquist component by 2 in even case
        xap[..., int(M / 2 + 1) - 1] = xap[..., int(M / 2 + 1) - 1] / 2
        xan[..., int(M / 2 + 1) - 1] = xan[..., int(M / 2 + 1) - 1] / 2
    else:
        xap[..., int((M + 3) / 2) - 1 : int(M + 1) + 1] = 0
        xan[..., int((M + 3) / 2) - 1 : int(M + 1) + 1] = 0

    # inverse Fourier transform along last axis
    xap = np.fft.ifft(xap)
    xan = np.fft.ifft(xan)

    # return central part plus half the mean
    xap = xap[..., int(N + 1) - 1 : int(2 * N + 1) - 1] + 0.5 * mx_
    xan = xan[..., int(N + 1) - 1 : int(2 * N + 1) - 1] + 0.5 * np.conj(mx_)

    if np.isrealobj(x):
        xa = xap + xan
    else:
        xa = (xap, xan)

    # return after reorganizing the axes
    if time_axis != -1 and time_axis != len(x.shape) - 1:
        return np.swapaxes(xa, time_axis, -1)
    else:
        return xa


def cartesian_to_rotary(
    ua: Union[np.ndarray, xr.DataArray],
    va: Union[np.ndarray, xr.DataArray],
    time_axis: Optional[int] = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotary signals (wp,wn) from analytic Cartesian signals (ua,va).

    If ua is the analytic signal from real-valued signal u, and va the analytic signal
    from real-valued signal v, then the positive (counterclockwise) and negative (clockwise)
    signals are defined by wp = 0.5*(up+1j*vp), wp = 0.5*(up-1j*vp).

    This function is the inverse of :func:`rotary_to_cartesian`.

    Parameters
    ----------
    ua : array_like
        Complex-valued analytic signal for first Cartesian component (zonal, east-west)
    va : array_like
        Complex-valued analytic signal for second Cartesian component (meridional, north-south)
    time_axis : int, optional
        The axis of the time array. Default is -1, which corresponds to the
        last axis.

    Returns
    -------
    wp : np.ndarray
        Complex-valued positive (counterclockwise) rotary signal.
    wn : np.ndarray
        Complex-valued negative (clockwise) rotary signal.

    Examples
    --------
    To obtain the rotary signals from a pair of real-valued signal:

    >>> u = np.random.rand(99)
    >>> v = np.random.rand(99)
    >>> wp, wn = cartesian_to_rotary(analytic_signal(u), analytic_signal(v))

    To specify that the time axis is along the first axis:

    >>> u = np.random.rand(100, 99)
    >>> v = np.random.rand(100, 99)
    >>> wp, wn = cartesian_to_rotary(analytic_signal(u), analytic_signal(v), time_axis=0)

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
        If the time axis is outside of the valid range ([-1, N-1]).

    References
    ----------
    Lilly JM, Olhede SC. 2010 Bivariate instantaneous frequency and bandwidth.
    IEEE T. Signal Proces. 58, 591–603. (10.1109/TSP.2009.2031729)

    See Also
    --------
    :func:`analytic_signal`, :func:`rotary_to_cartesian`
    """
    # u and v arrays must have the same shape.
    if not ua.shape == va.shape:
        raise ValueError("u and v must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(ua.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(ua.shape) - 1}])."
        )

    wp = 0.5 * (ua + 1j * va)
    wn = 0.5 * (ua - 1j * va)

    return wp, wn


def ellipse_parameters(
    xa: Union[np.ndarray, xr.DataArray],
    ya: Union[np.ndarray, xr.DataArray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the instantaneous parameters of a modulated elliptical signal from its analytic Cartesian signals.

    Parameters
    ----------
    xa : array_like
        Complex-valued analytic signal for first Cartesian component (zonal, east-west).
    ya : array_like
        Complex-valued analytic signal for second Cartesian component (meridional, north-south).

    Returns
    -------
    kappa : np.ndarray
        Ellipse root-mean-square amplitude.
    lambda : np.ndarray
        Ellipse linearity between -1 and 1, or departure from circular motion (lambda=0).
    theta : np.ndarray
        Ellipse orientation in radian.
    phi : np.ndarray
        Ellipse phase in radian.

    Examples
    --------

    To obtain the ellipse parameters from a pair of real-valued signals (x, y):

    >>> kappa, lambda, theta, phi = ellipse_parameters(analytic_signal(x), analytic_signal(y))

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.

    References
    ----------
    Lilly JM, Olhede SC. 2010 Bivariate instantaneous frequency and bandwidth.
    IEEE T. Signal Proces. 58, 591–603. (10.1109/TSP.2009.2031729).

    See Also
    --------
    :func:`modulated_ellipse_signal`, :func:`analytic_signal`, :func:`rotary_to_cartesian`, :func:`cartesian_to_rotary`

    """

    # u and v arrays must have the same shape.
    if not xa.shape == ya.shape:
        raise ValueError("xa and ya must have the same shape.")

    X = np.abs(xa)
    Y = np.abs(ya)
    phix = np.angle(xa)
    phiy = np.angle(ya)

    phia = 0.5 * (phix + phiy + 0.5 * np.pi)
    phid = 0.5 * (phix - phiy - 0.5 * np.pi)

    P = 0.5 * np.sqrt(X**2 + Y**2 + 2 * X * Y * np.cos(2 * phid))
    N = 0.5 * np.sqrt(X**2 + Y**2 - 2 * X * Y * np.cos(2 * phid))

    phip = np.unwrap(
        phia
        + np.unwrap(np.imag(np.log(X * np.exp(1j * phid) + Y * np.exp(-1j * phid))))
    )
    phin = np.unwrap(
        phia
        + np.unwrap(np.imag(np.log(X * np.exp(1j * phid) - Y * np.exp(-1j * phid))))
    )

    kappa = np.sqrt(P**2 + N**2)
    lambda_ = (2 * P * N * np.sign(P - N)) / (P**2 + N**2)

    # For vanishing linearity, put in very small number to have sign information
    lambda_[lambda_ == 0] = np.sign(P[lambda_ == 0] - N[lambda_ == 0]) * (1e-12)

    theta = np.unwrap(0.5 * (phip - phin))
    phi = np.unwrap(0.5 * (phip + phin))

    lambda_ = np.real(lambda_)

    return kappa, lambda_, theta, phi


def modulated_ellipse_signal(
    kappa: Union[np.ndarray, xr.DataArray],
    lambda_: Union[np.ndarray, xr.DataArray],
    theta: Union[np.ndarray, xr.DataArray],
    phi: Union[np.ndarray, xr.DataArray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the analytic Cartesian signals (xa, ya) from the instantaneous parameters of a modulated elliptical signal.

    This function is the inverse of :func:`ellipse_parameters`.

    Parameters
    ----------
    kappa : array_like
        Ellipse root-mean-square amplitude.
    lambda : array_like
        Ellipse linearity between -1 and 1, or departure from circular motion (lambda=0).
    theta : array_like
        Ellipse orientation in radian.
    phi : array_like
        Ellipse phase in radian.
    time_axis : int, optional
        The axis of the time array. Default is -1, which corresponds to the
        last axis.

    Returns
    -------
    xa : np.ndarray
        Complex-valued analytic signal for first Cartesian component (zonal, east-west).
    ya : np.ndarray
        Complex-valued analytic signal for second Cartesian component (meridional, north-south).

    Examples
    --------

    To obtain the analytic signals from the instantaneous parameters of a modulated elliptical signal:

    >>> xa, ya = modulated_ellipse_signal(kappa, lambda, theta, phi)

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.

    References
    ----------
    Lilly JM, Olhede SC. 2010 Bivariate instantaneous frequency and bandwidth.
    IEEE T. Signal Proces. 58, 591–603. (10.1109/TSP.2009.2031729).

    See Also
    --------
    :func:`ellipse_parameters`, :func:`analytic_signal`, :func:`rotary_to_cartesian`, :func:`cartesian_to_rotary`

    """

    # make sure all input arrays have the same shape
    if not kappa.shape == lambda_.shape == theta.shape == phi.shape:
        raise ValueError("All input arrays must have the same shape.")

    # calculate semi major and semi minor axes
    a = kappa * np.sqrt(1 + np.abs(lambda_))
    b = np.sign(lambda_) * kappa * np.sqrt(1 - np.abs(lambda_))

    # define b to be positive for lambda exactly zero
    b[lambda_ == 0] = kappa[lambda_ == 0]

    xa = np.exp(1j * phi) * (a * np.cos(theta) + 1j * b * np.sin(theta))
    ya = np.exp(1j * phi) * (a * np.sin(theta) - 1j * b * np.cos(theta))

    mask = np.isinf(kappa * lambda_ * theta * phi)
    xa[mask] = np.inf + 1j * np.inf
    ya[mask] = np.inf + 1j * np.inf

    return xa, ya


def rotary_to_cartesian(
    wp: Union[np.ndarray, xr.DataArray],
    wn: Union[np.ndarray, xr.DataArray],
    time_axis: Optional[int] = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Cartesian analytic signals (ua, va) from rotary signals (wp, wn)
    as ua = wp + wn and va = -1j * (wp - wn).

    This function is the inverse of :func:`cartesian_to_rotary`.

    Parameters
    ----------
    wp : array_like
        Complex-valued positive (counterclockwise) rotary signal.
    wn : array_like
        Complex-valued negative (clockwise) rotary signal.
    time_axis : int, optional
        The axis of the time array. Default is -1, which corresponds to the
        last axis.

    Returns
    -------
    ua : array_like
        Complex-valued analytic signal, first Cartesian component (zonal, east-west)
    va : array_like
        Complex-valued analytic signal, second Cartesian component (meridional, north-south)

    Examples
    --------

    To obtain the Cartesian analytic signals from a pair of rotary signals (wp,wn):

    >>> ua, va = rotary_to_cartesian(wp, wn)

    To specify that the time axis is along the first axis:

    >>> ua, va = rotary_to_cartesian(wp, wn, time_axis=0)

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
        If the time axis is outside of the valid range ([-1, N-1]).

    References
    ----------
    Lilly JM, Olhede SC. 2010 Bivariate instantaneous frequency and bandwidth.
    IEEE T. Signal Proces. 58, 591–603. (10.1109/TSP.2009.2031729)

    See Also
    --------
    :func:`analytic_signal`, :func:`cartesian_to_rotary`
    """

    if not wp.shape == wn.shape:
        raise ValueError("u and v must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(wp.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(wp.shape) - 1}])."
        )

    # I think this may return xarray dataarrays if that's the input
    ua = wp + wn
    va = -1j * (wp - wn)

    return ua, va

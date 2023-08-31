"""
This module provides wavelet functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import warnings
from math import gamma, lgamma


def wavetrans(
    x: np.ndarray,
    psi: np.ndarray,
    norm: Optional[str] = "bandpass",
    boundary: Optional[str] = "mirror",
    time_axis: Optional[int] = -1,
    freq_axis: Optional[int] = -2,
    order_axis: Optional[int] = -3,
) -> np.ndarray:
    """
    Continuous wavelet transform.

    Parameters
    ----------
    x : np.ndarray
        Real- or complex-valued signals
    psi : np.ndarray
        A suite of Morse wavelets as returned by function morsewave. The dimensions
        of the suite of Morse wavelets are typically (f_order, freq_axis, time_axis).
        The time axis of the wavelets must be the last one and matches the length of the time axis of x.
        The normalization of the wavelets is assumed to be "bandpassed", if not use kwarg norm="energy".
    boundary : str, optional
        The boundary condition to be imposed at the edges of the time series.
        Allowed values are "mirror", "zeros", and "periodic".
        Default is "mirror".
    order_axis : int, optional
        Axis of psi for the order of the wavelets (default is first or 0)
    freq_axis : int, optional
        Axis of psi for the frequencies of the wavelet (default is second or 1)
    time_axis : int, optional
        Axis on which the time is defined for x (default is last, or -1). The time axis of the
        wavelets must be last.

    Returns
    -------
    wt : np.ndarray
        Time-domain wavelet transforms. w shape will be ((series_orders), order, freq_axis, time_axis).

    Examples
    --------
    To write.

    Raises
    ------
    ValueError
        If the time axis is outside of the valid range ([-1, N-1]).
        If the shape of time axis is different for input and wavelet.
        If ``boundary not in ["mirror", "zeros", "periodic"]``.

    See Also
    --------
    :func:`morsewave`, `morsefreq`
    """
    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(x.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )
    # Positions and time arrays must have the same shape.
    if x.shape[time_axis] != psi.shape[-1]:
        raise ValueError("x and psi time axes must have the same length.")

    psi_ = np.moveaxis(psi, [freq_axis, order_axis], [-2, -3])

    # initialization: output will be ((x_orders),f_order, freq_axis, time_axis)
    # w = np.tile(
    #     np.expand_dims(np.zeros_like(x), (-3, -2)),
    #     (1, np.shape(psi)[-3], np.shape(psi)[-2], 1),
    # )

    # if x is of dimension 1 we need to expand
    # otherwise make sure time axis is last
    if np.ndim(x) < 2:
        x_ = np.expand_dims(x, axis=0)
    else:
        x_ = np.moveaxis(x, time_axis, -1)

    if ~np.all(np.isreal(x)):
        if norm == "energy":
            x_ = x_ / np.sqrt(2)
        elif norm == "bandpass":
            x_ = x_ / 2

    # to do: add detrending option by default?

    # apply boundary conditions
    if boundary == "mirror":
        x_ = np.concatenate((np.flip(x_, axis=-1), x_, np.flip(x_, axis=-1)), axis=-1)
    elif boundary == "zeros":
        x_ = np.concatenate((np.zeros_like(x_), x_, np.zeros_like(x_)), axis=-1)
    elif boundary == "periodic":
        x_ = np.concatenate((x_, x_, x_), axis=-1)
    else:
        raise ValueError("boundary must be one of 'mirror', 'align', or 'zeros'.")

    # danger zone, use different letters from jlab: n is their time length
    time_length = np.shape(x)[-1]
    time_length_ = np.shape(x_)[-1]

    # pad wavelet with zeros?
    # psi_ = np.concatenate((np.zeros_like(psi), psi, np.zeros_like(psi)), axis=-1)
    order_length, freq_length, _ = np.shape(psi)
    psi_ = np.zeros((order_length, freq_length, time_length_), dtype="cfloat")
    # display(np.shape(psi_))
    index = slice(
        int(0 + np.floor(time_length_ - time_length) / 2),
        int(time_length + np.floor(time_length_ - time_length) / 2),
    )
    psi_[:, :, index] = psi

    # here we assume that wavelet and data have exactly the same number of data points;
    # to do: include case wavelet is shorter or longer

    # take fft along axis = -1
    psif_ = np.fft.fft(psi_)
    om = 2 * np.pi * np.linspace(0, 1 - 1 / time_length_, time_length_)
    if time_length_ % 2 == 0:
        psif_ = psif_ * np.exp(1j * -om * (time_length_ + 1) / 2) * np.sign(np.pi - om)
    else:
        psif_ = psif_ * np.exp(1j * -om * (time_length_ + 1) / 2)

    # here I should be able to automate the tiling without assuming extra dimensions of psi
    X_ = np.tile(
        np.expand_dims(np.fft.fft(x_), (-3, -2)),
        (1, order_length, freq_length, 1),
    )
    # finally the transform
    wt = np.fft.ifft(X_ * np.conj(psif_))
    # return central part
    wt = wt[..., index]

    # if I comment this the complex test passes but not the boundary test
    # and vice versa
    # I prefer the complex test to pass for now
    # not sure I understand if this is needed
    # if np.all(np.isreal(x_)):
    #    w = np.real(w)
    # should we squeeze w? probably
    wt = np.squeeze(wt)

    return wt


def morsewave(
    n: int,
    ga: float,
    be: float,
    fs: np.ndarray,
    order: Optional[int] = 1,
    norm: Optional[str] = "bandpass",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Morse wavelets of Olhede and Walden (2002).

    Parameters
    ----------
    n: int
       Length of the wavelet.
    ga: float
       Gamma parameter of the wavelet.
    be: float
       Beta parameter of the wavelet.
    fs: np.ndarray
       The radian frequencies at which the Fourier transform of the wavelets
       reach their maximum amplitudes. fs is between 0 and 2 * np.pi * 0.5,
       the normalized Nyquist radian frequency.
    order: int
        Wavelet order, default is 1.
    norm:  str, optional
       Normalization for the wavelets. Default is "bandpass".
       "bandpass" uses "bandpass normalization", meaning that the FFT of the wavelet
       has a peak value of 2 for all frequencies fs. "energy" uses the unit
       energy normalization. The time-domain wavelet energy np.sum(np.abs(psi)**2)
       is then always unity.

    Returns
    -------
    psi : np.ndarray
        Time-domain wavelets. psi will be of shape (n,np.size(fs),k).
    psif: np.ndarray
        Frequency-domain wavelets. psif will be of shape (n,np.size(fs),k).

    Examples
    --------
    To write.

    See Also
    --------
    :func:`wavetrans`, `morsefreq`
    """
    # add a test for fs being a numpy array
    # initialization
    psi = np.zeros((n, order, len(fs)), dtype="cdouble")
    psif = np.zeros((n, order, len(fs)), dtype="cdouble")

    # call to morsewave take only ga and be as float, no array
    fo, _, _ = morsefreq(ga, be)
    for i in range(0, len(fs)):
        psi_tmp = np.zeros((n, order), dtype="cdouble")
        psif_tmp = np.zeros((n, order), dtype="cdouble")

        # wavelet frequencies
        fact = np.abs(fs[i]) / fo
        # om first dim is n points
        om = 2 * np.pi * np.linspace(0, 1 - 1 / n, n) / fact
        if norm == "energy":
            if be == 0:
                psizero = np.exp(-(om**ga))
            else:
                psizero = np.exp(be * np.log(om) - om**ga)
            # psizero = np.where(be == 0, psizero0, psizero1)
        elif norm == "bandpass":
            if be == 0:
                psizero = 2 * np.exp(-(om**ga))
            else:
                psizero = 2 * np.exp(
                    -be * np.log(fo) + fo**ga + be * np.log(om) - om**ga
                )
            # psizero = np.where(be == 0, psizero0, psizero1)
        else:
            raise ValueError(
                "Normalization option (norm) must be one of 'energy' or 'bandpass'."
            )
        psizero[0] = 0.5 * psizero[0]
        # Replace NaN with zeros in psizero
        psizero = np.nan_to_num(psizero, copy=False, nan=0.0)
        # to do, derive second family wavelet, here do first family
        # spectral domain wavelet
        psif_tmp = _morsewave_first_family(fact, n, ga, be, om, psizero, order=order, norm=norm)
        psif_tmp = np.nan_to_num(psif_tmp, posinf=0, neginf=0)
        # shape of psif_tmp is points, order
        # center wavelet
        ommat = np.tile(np.expand_dims(om, -1), (order))
        psif_tmp = psif_tmp * np.exp(1j * ommat * (n + 1) / 2 * fact)
        # time domain wavelet
        psi_tmp = np.fft.ifft(psif_tmp, axis=0)
        if fs[i] < 0:
            psi[:, :, i] = np.conj(psi_tmp)
            psif_tmp[1:-1, :] = np.flip(psif_tmp[1:-1, :], axis=0)
            psif[:, :, i] = psif_tmp
        else:
            psif[:, :, i] = psif_tmp
            psi[:, :, i] = psi_tmp

    # reorder dimension to be (order, frequency, time steps)
    # enforce length 1 for first axis is order=1 (no squeezing)
    psi = np.moveaxis(psi, [0, 1, 2], [2, 0, 1])
    psif = np.moveaxis(psif, [0, 1, 2], [2, 0, 1])
    return psi, psif


def _morsewave_first_family(
    fact: float,
    n: int,
    ga: float,
    be: float,
    om: np.ndarray,
    psizero: np.ndarray,
    order: Optional[int] = 1,
    norm: Optional[str] = "bandpass",
) -> np.ndarray:
    """
    Derive first family wavelet
    """
    r = (2 * be + 1) / ga
    c = r - 1
    L = np.zeros_like(om, dtype="float")
    psif1 = np.zeros((np.shape(psizero)[0], order))

    for i in np.arange(0, order):
        if norm == "energy":
            A = morseafun(ga, be, order=i + 1, norm=norm)
            coeff = np.sqrt(1 / fact) * A
        elif norm == "bandpass":
            if be != 0:
                coeff = np.sqrt(np.exp(_lgamma(r) + _lgamma(i + 1) - _lgamma(i + r)))
            else:
                coeff = 1

        index = slice(0, int(np.round(n / 2)))  # how to define indices?
        L[index] = _laguerre(2 * om[index] ** ga, i, c)
        psif1[:, i] = coeff * psizero * L

    return psif1


def morsefreq(
    ga: Union[np.ndarray, float],
    be: Union[np.ndarray, float],
) -> Union[Tuple[np.ndarray], Tuple[float]]:
    """
    Frequency measures for generalized Morse wavelets. This functions calculates
    three different measures of the frequency of the lowest-order generalized Morse
    wavelet specified by parameters gamma (ga) and beta (beta).

    Note that all frequency quantities here are *radian* as in cos(omega t)and not
    cyclic as in np.cos(2 np.pi f t).

    For be=0, the "wavelet" becomes an analytic lowpass filter, and fm
    is not defined in the usual way.  Instead, fm is defined as the point
    at which the filter has decayed to one-half of its peak power.

    For details see Lilly and Olhede (2009).  Higher-order properties of analytic
    wavelets.  IEEE Trans. Sig. Proc., 57 (1), 146--160.

    Parameters
    ----------
    ga: np.ndarray or float
       Gamma parameter of the wavelet.
    be: np.ndarray or float
       Beta parameter of the wavelet.

    Returns
    -------
    fm: np.ndarray
        The modal or peak frequency.
    fe: np.ndarray
        The "energy" frequency.
    fi: np.ndarray
        The instantaneous frequency at the wavelet center.

    Examples
    --------
    To write.

    See Also
    --------
    :func:`morsewave`
    """
    # add test for type and shape in case of ndarray
    fm = np.where(
        be == 0, np.log(2) ** (1 / ga), np.exp((1 / ga) * (np.log(be) - np.log(ga)))
    )

    fe = 1 / (2 ** (1 / ga)) * _gamma((2 * be + 2) / ga) / _gamma((2 * be + 1) / ga)

    fi = _gamma((be + 2) / ga) / _gamma((be + 1) / ga)

    return fm, fe, fi


def morsespace(
    ga: Union[np.ndarray, float],
    be: Union[np.ndarray, float],
    n: int,
    low: Optional[float] = 5,
    high: Optional[float] = 0.1,
    d: Optional[int] = 4,
    eta: Optional[float] = 0.1,
) -> np.ndarray:
    """
    Returns logarithmically-spaced frequencies for generalized Morse wavelets.

    Parameters
    ----------
    ga: np.ndarray or float
       Gamma parameter of the wavelet.
    be: np.ndarray or float
       Beta parameter of the wavelet.
    n:
    high:
    low:
    d:
    eta:

    Returns
    -------
    fs: np.ndarray

    """
    p, _, _ = morseprops(ga, be)

    _high = _morsehigh(ga, be, high)
    high_ = np.min(np.append(_high, np.pi))

    _low = 2 * np.sqrt(2) * p * low / n
    low_ = np.max(np.append(_low, 0))

    r = 1 + 1 / (d * p)
    m = np.floor(np.log10(high_ / low_) / np.log10(r))
    fs = high_ * np.ones(int(m + 1)) / r ** np.arange(0, m + 1)

    return fs


def _morsehigh(
    ga: Union[np.ndarray, float],
    be: Union[np.ndarray, float],
    eta: float,
) -> Union[np.ndarray, float]:
    """
    High-frequency cutoff of the generalized Morse wavelets.
    ga and be should be of the same length.
    """
    m = 10000
    omhigh = np.linspace(0, np.pi, m)
    f = np.zeros_like(ga, dtype="float")

    for i in range(0, len(ga)):
        fm, _, _ = morsefreq(ga[i], be[i])
        om = fm * np.pi / omhigh
        lnpsi1 = be[i] / ga[i] * np.log(np.exp(1) * ga[i] / be[i])
        lnpsi2 = be[i] * np.log(om) - om ** ga[i]
        lnpsi = lnpsi1 + lnpsi2
        index = np.nonzero(np.log(eta) - lnpsi < 0)[0][0]
        f[i] = omhigh[index]

    return f


def morseprops(
    ga: Union[np.ndarray, float],
    be: Union[np.ndarray, float],
) -> Union[Tuple[np.ndarray], Tuple[float]]:
    """
    Properties of the demodulated generalized Morse wavelets.

    Parameters
    ----------
    ga: np.ndarray or float
       Gamma parameter of the wavelet.
    be: np.ndarray or float
       Beta parameter of the wavelet.

    Returns
    -------
    p: np.ndarray or float
    skew: np.ndarray or float
    kurt: np.ndarray or float
    """
    # test common size? or could be broadcasted
    p = np.sqrt(ga * be)
    skew = ga - 3 / p
    kurt = 3 - skew**2 - 2 / p**2

    return p, skew, kurt


def morseafun(
    ga: Union[np.ndarray, float],
    be: Union[np.ndarray, float],
    order: Optional[np.int64] = 1,
    norm: Optional[str] = "bandpass",
) -> float:
    # add test for type and shape in case of ndarray
    if norm == "energy":
        r = (2 * be + 1) / ga
        a = (2 * np.pi * ga * (2**r) * np.exp(_lgamma(order) - _lgamma(order + r - 1))) ** 0.5
    elif norm == "bandpass":
        om, _, _ = morsefreq(ga, be)
        a = np.where(be == 0, 2, 2 / (np.exp(be * np.log(om) - om**ga)))
    else:
        raise ValueError(
            "Normalization option (norm) must be one of 'energy' or 'bandpass'."
        )

    return a


def _gamma(
    x: Union[np.ndarray, float],
) -> np.ndarray:
    """
    Returns gamma function values. Wrapper for math.gamma which is
    needed for array inputs.
    """
    # add test for type and shape in case of ndarray?
    if type(x) is np.ndarray:
        x_ = x.flatten()
        y = np.zeros_like(x_, dtype="float")
        for i in range(0, np.size(x)):
            y[i] = gamma(x_[i])
        y = np.reshape(y, np.shape(x))
    else:
        y = gamma(x)

    return y


# this maybe not useful
def _lgamma(
    x: Union[np.ndarray, float],
) -> np.ndarray:
    """
    Returns logarithm of gamma function values. Wrapper for math.lgamma which is
    needed for array inputs.
    """
    # add test for type and shape in case of ndarray?
    if type(x) is np.ndarray:
        x_ = x.flatten()
        y = np.zeros_like(x_, dtype="float")
        for i in range(0, np.size(x)):
            y[i] = lgamma(x_[i])
        y = np.reshape(y, np.shape(x))
    else:
        y = lgamma(x)

    return y


def _laguerre(
    x: Union[np.ndarray, float],
    k: float,
    c: float,
) -> np.ndarray:
    """
    Generalized Laguerre polynomials
    """
    y = np.zeros_like(x, dtype="float")
    for i in np.arange(0, k + 1):
        fact = np.exp(_lgamma(k + c + 1) - _lgamma(c + i + 1) - _lgamma(k - i + 1))
        y = y + (-1) ** i * fact * x**i / _gamma(i + 1)
    return y

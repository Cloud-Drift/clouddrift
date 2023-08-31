"""
This module provides wavelet functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import warnings
from scipy.special import gamma as _gamma, gammaln as _lgamma


def wavetrans(
    x: np.ndarray,
    wave: np.ndarray,
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
    wave : np.ndarray
        A suite of Morse wavelets as returned by function morsewave. The dimensions
        of the suite of Morse wavelets are typically (f_order, freq_axis, time_axis).
        The time axis of the wavelets must be the last one and matches the length of the time axis of x.
        The normalization of the wavelets is assumed to be "bandpassed", if not use kwarg norm="energy".
    boundary : str, optional
        The boundary condition to be imposed at the edges of the time series.
        Allowed values are "mirror", "zeros", and "periodic".
        Default is "mirror".
    order_axis : int, optional
        Axis of wave for the order of the wavelets (default is first or 0)
    freq_axis : int, optional
        Axis of wave for the frequencies of the wavelet (default is second or 1)
    time_axis : int, optional
        Axis on which the time is defined for x (default is last, or -1). The time axis of the
        wavelets must be last.

    Returns
    -------
    wtx : np.ndarray
        Time-domain wavelet transforms of input x. w shape will be ((series_orders), order, freq_axis, time_axis).

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
    if x.shape[time_axis] != wave.shape[-1]:
        raise ValueError("x and wave time axes must have the same length.")

    wave_ = np.moveaxis(wave, [freq_axis, order_axis], [-2, -3])

    # if x is of dimension 1 we need to expand
    # otherwise make sure time axis is last
    if np.ndim(x) < 2:
        x_ = np.expand_dims(x, axis=0)
    else:
        x_ = np.moveaxis(x, time_axis, -1)

    if ~np.all(np.isreal(x)):
        if norm == "energy":
            x_ /= np.sqrt(2)
        elif norm == "bandpass":
            x_ /= 2

    # to do: add detrending option by default

    # apply boundary conditions
    if boundary == "mirror":
        x_ = np.concatenate((np.flip(x_, axis=-1), x_, np.flip(x_, axis=-1)), axis=-1)
    elif boundary == "zeros":
        x_ = np.concatenate((np.zeros_like(x_), x_, np.zeros_like(x_)), axis=-1)
    elif boundary == "periodic":  # JML: this not needed in this case you just want n
        x_ = np.concatenate((x_, x_, x_), axis=-1)
    else:
        raise ValueError("boundary must be one of 'mirror', 'align', or 'zeros'.")

    time_length = np.shape(x)[-1]
    time_length_ = np.shape(x_)[-1]

    # pad wavelet with zeros: JML ok
    order_length, freq_length, _ = np.shape(wave)
    _wave = np.zeros((order_length, freq_length, time_length_), dtype=np.cdouble)

    index = slice(
        int(np.floor(time_length_ - time_length) / 2),
        int(time_length + np.floor(time_length_ - time_length) / 2),
    )
    _wave[:, :, index] = wave_

    # take fft along axis = -1
    _wavefft = np.fft.fft(_wave)
    om = 2 * np.pi * np.linspace(0, 1 - 1 / time_length_, time_length_)
    if time_length_ % 2 == 0:
        _wavefft = (
            _wavefft * np.exp(1j * -om * (time_length_ + 1) / 2) * np.sign(np.pi - om)
        )
    else:
        _wavefft = _wavefft * np.exp(1j * -om * (time_length_ + 1) / 2)

    # here we should be able to automate the tiling without assuming extra dimensions of wave
    X_ = np.tile(
        np.expand_dims(np.fft.fft(x_), (-3, -2)),
        (1, order_length, freq_length, 1),
    )

    # finally the transform; return precision of input `x``; central part only
    complex_dtype = np.cdouble if x.dtype == np.single else np.csingle
    wtx = np.fft.ifft(X_ * np.conj(_wavefft)).astype(complex_dtype)
    wtx = wtx[..., index]
    # remove extra dimensions
    wtx = np.squeeze(wtx)

    return wtx


def morsewave(
    n: int,
    ga: float,
    be: float,
    rad_freq: np.ndarray,
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
    rad_freq: np.ndarray
       The radian frequencies at which the Fourier transform of the wavelets
       reach their maximum amplitudes. rad_freq is between 0 and 2 * np.pi * 0.5,
       the normalized Nyquist radian frequency.
    order: int
        Wavelet order, default is 1.
    norm:  str, optional
       Normalization for the wavelets. Default is "bandpass".
       "bandpass" uses "bandpass normalization", meaning that the FFT of the wavelet
       has a peak value of 2 for all frequencies rad_freq. "energy" uses the unit
       energy normalization. The time-domain wavelet energy np.sum(np.abs(wave)**2)
       is then always unity.

    Returns
    -------
    wave : np.ndarray
        Time-domain wavelets. wave will be of shape (n,np.size(rad_freq),k).
    wavefft: np.ndarray
        Frequency-domain wavelets. wavefft will be of shape (n,np.size(rad_freq),k).

    Examples
    --------
    To write.

    See Also
    --------
    :func:`wavetrans`, `morsefreq`
    """
    # add a test for rad_freq being a numpy array
    # initialization
    wave = np.zeros((n, order, len(rad_freq)), dtype=np.cdouble)
    wavefft = np.zeros((n, order, len(rad_freq)), dtype=np.cdouble)

    # call to morsewave take only ga and be as float, no array
    fo, _, _ = morsefreq(ga, be)
    for i in range(len(rad_freq)):
        wave_tmp = np.zeros((n, order), dtype=np.cdouble)
        wavefft_tmp = np.zeros((n, order), dtype=np.cdouble)

        # wavelet frequencies
        fact = np.abs(rad_freq[i]) / fo
        # norm_rad_freq first dim is n points
        norm_rad_freq = 2 * np.pi * np.linspace(0, 1 - 1 / n, n) / fact
        if norm == "energy":
            if be == 0:
                wavezero = np.exp(-(norm_rad_freq**ga))
            else:
                wavezero = np.exp(be * np.log(norm_rad_freq) - norm_rad_freq**ga)
        elif norm == "bandpass":
            if be == 0:
                wavezero = 2 * np.exp(-(norm_rad_freq**ga))
            else:
                wavezero = 2 * np.exp(
                    -be * np.log(fo)
                    + fo**ga
                    + be * np.log(norm_rad_freq)
                    - norm_rad_freq**ga
                )
        else:
            raise ValueError(
                "Normalization option (norm) must be one of 'energy' or 'bandpass'."
            )
        wavezero[0] = 0.5 * wavezero[0]
        # Replace NaN with zeros in wavezero
        wavezero = np.nan_to_num(wavezero, copy=False, nan=0.0)
        # second family is never used
        wavefft_tmp = _morsewave_first_family(
            fact, n, ga, be, norm_rad_freq, wavezero, order=order, norm=norm
        )
        wavefft_tmp = np.nan_to_num(wavefft_tmp, posinf=0, neginf=0)
        # shape of wavefft_tmp is points, order
        # center wavelet
        norm_rad_freq_mat = np.tile(np.expand_dims(norm_rad_freq, -1), (order))
        wavefft_tmp = wavefft_tmp * np.exp(1j * norm_rad_freq_mat * (n + 1) / 2 * fact)
        # time domain wavelet
        wave_tmp = np.fft.ifft(wavefft_tmp, axis=0)
        if rad_freq[i] < 0:
            wave[:, :, i] = np.conj(wave_tmp)
            wavefft_tmp[1:-1, :] = np.flip(wavefft_tmp[1:-1, :], axis=0)
            wavefft[:, :, i] = wavefft_tmp
        else:
            wavefft[:, :, i] = wavefft_tmp
            wave[:, :, i] = wave_tmp

    # reorder dimension to be (order, frequency, time steps)
    # enforce length 1 for first axis if order=1 (no squeezing)
    wave = np.moveaxis(wave, [0, 1, 2], [2, 0, 1])
    wavefft = np.moveaxis(wavefft, [0, 1, 2], [2, 0, 1])
    return wave, wavefft


def _morsewave_first_family(
    fact: float,
    n: int,
    ga: float,
    be: float,
    norm_rad_freq: np.ndarray,
    wavezero: np.ndarray,
    order: Optional[int] = 1,
    norm: Optional[str] = "bandpass",
) -> np.ndarray:
    """
    Derive first family wavelet. Internal use only.
    """
    r = (2 * be + 1) / ga
    c = r - 1
    L = np.zeros_like(norm_rad_freq, dtype=np.float64)
    wavefft1 = np.zeros((np.shape(wavezero)[0], order))

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
        L[index] = _laguerre(2 * norm_rad_freq[index] ** ga, i, c)
        wavefft1[:, i] = coeff * wavezero * L

    return wavefft1


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
    rad_freq: np.ndarray

    Examples
    --------
    To write.

    """
    p, _, _ = morseprops(ga, be)

    _high = _morsehigh(ga, be, high)
    high_ = np.min(np.append(_high, np.pi))

    _low = 2 * np.sqrt(2) * p * low / n
    low_ = np.max(np.append(_low, 0))

    r = 1 + 1 / (d * p)
    m = np.floor(np.log10(high_ / low_) / np.log10(r))
    rad_freq = high_ * np.ones(int(m + 1)) / r ** np.arange(0, m + 1)

    return rad_freq


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
        lnwave1 = be[i] / ga[i] * np.log(np.exp(1) * ga[i] / be[i])
        lnwave2 = be[i] * np.log(om) - om ** ga[i]
        lnwave = lnwave1 + lnwave2
        index = np.nonzero(np.log(eta) - lnwave < 0)[0][0]
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
        a = (
            2 * np.pi * ga * (2**r) * np.exp(_lgamma(order) - _lgamma(order + r - 1))
        ) ** 0.5
    elif norm == "bandpass":
        fm, _, _ = morsefreq(ga, be)
        a = np.where(be == 0, 2, 2 / (np.exp(be * np.log(fm) - fm**ga)))
    else:
        raise ValueError(
            "Normalization option (norm) must be one of 'energy' or 'bandpass'."
        )

    return a


# def _gamma(
#    x: Union[np.ndarray, float],
# ) -> np.ndarray:
#    """
#    Returns gamma function values. Wrapper for math.gamma which is
#    needed for array inputs.
#    """
#    # add test for type and shape in case of ndarray?
#    if type(x) is np.ndarray:
#        x_ = x.flatten()
#        y = np.zeros_like(x_, dtype="float")
#        for i in range(0, np.size(x)):
#            y[i] = math.gamma(x_[i])
#        y = np.reshape(y, np.shape(x))
#    else:
#        y = math.gamma(x)
#
#    return y


# this maybe not useful
# def _lgamma(
#    x: Union[np.ndarray, float],
# ) -> np.ndarray:
#    """
#    Returns logarithm of gamma function values. Wrapper for math.lgamma which is
#    needed for array inputs.
#    """
#    # add test for type and shape in case of ndarray?
#    if type(x) is np.ndarray:
#        x_ = x.flatten()
#        y = np.zeros_like(x_, dtype="float")
#        for i in range(0, np.size(x)):
#            y[i] = math.lgamma(x_[i])
#        y = np.reshape(y, np.shape(x))
#    else:
#        y = math.lgamma(x)
#
#    return y


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

"""
This module provides wavelet functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import warnings
from scipy.special import gamma as _gamma, gammaln as _lgamma


def wavelet_transform(
    x: np.ndarray,
    wavelet: np.ndarray,
    normalization: Optional[str] = "bandpass",
    boundary: Optional[str] = "mirror",
    time_axis: Optional[int] = -1,
    freq_axis: Optional[int] = -2,
    order_axis: Optional[int] = -3,
) -> np.ndarray:
    """
    Apply a continuous wavelet transform to an input signal using an input wavelet
    function. Such wavelet can be provided by the function `morse_wavelet`.

    Parameters
    ----------
    x : np.ndarray
        Real- or complex-valued signals.
    wavelet : np.ndarray
        A suite of time-domain wavelets, typically returned by the function ``morse_wavelet``.
        The length of the time axis of the wavelets must be the last one and matches the
        length of the time axis of x. The other dimensions (axes) of the wavelets (orders and frequencies) are
        typically organized as orders, frequencies, and time, unless specified by optional arguments freq_axis and order_axis.
        The normalization of the wavelets is assumed to be "bandpass", if not use kwarg normalization="energy", see ``morse_wavelet``.
    normalization: str, optional
       Normalization for the ``wavelet`` input. By default it is assumed to be ``"bandpass"``
       which uses a bandpass normalization, meaning that the FFT of the wavelets
       have peak value of 2 for all central frequencies. The other option is ``"energy"``
       which uses the unit energy normalization. In this last case the time-domain wavelet
       energies ``np.sum(np.abs(wave)**2)`` are always unity. See ``morse_wavelet``.
    boundary : str, optional
        The boundary condition to be imposed at the edges of the input signal ``x``.
        Allowed values are ``"mirror"``, ``"zeros"``, and ``"periodic"``. Default is ``"mirror"``.
    time_axis : int, optional
        Axis on which the time is defined for input ``x`` (default is last, or -1). Note that the time axis of the
        wavelets must be last.
    freq_axis : int, optional
        Axis of ``wavelet`` for the frequencies (default is second or 1)
    order_axis : int, optional
        Axis of ``wavelet`` for the orders (default is first or 0)

    Returns
    -------
    wtx : np.ndarray
        Time-domain wavelet transform of input ``x``. The axes of ``wtx`` will be organized as (x axes), orders, frequencies, time
        unless ``time_axis`` is different from last (-1) in which case it will be moved back to its original position within the axes of ``x``.

    Examples
    --------

    Apply a wavelet transform with a Morse wavelet with gamma parameter 3, beta parameter 4, at radian frequency 0.2 cycles per unit time:

    >>> x = np.random.random(1024)
    >>> wavelet, _ = morse_wavelet(1024, 3, 4, np.array([2*np.pi*0.2]))
    >>> wtx = wavelet_transform(x, wavelet)

    The input signal can have an arbitrary number of dimensions but its ``time_axis`` must be specified if it is not the last:

    >>> x = np.random.random((1024,10,15))
    >>> wavelet, _ = morse_wavelet(1024, 3, 4, np.array([2*np.pi*0.2]))
    >>> wtx = wavelet_transform(x, wavelet,time_axis=0)

    If the wavelet was generated with ``morse_wavelet`` with the ``"energy"`` normalization, this must be specified with ``wavelet_transform`` as well:
    >>> x = np.random.random(1024)
    >>> wavelet, _ = morse_wavelet(1024, 3, 4, np.array([2*np.pi*0.2]), normalization="energy")
    >>> wtx = wavelet_transform(x, wavelet, normalization="energy")

    Raises
    ------
    ValueError
        If the time axis is outside of the valid range ([-1, N-1]).
        If the shape of time axis is different for input signal and wavelet.
        If boundary optional argument is not in ["mirror", "zeros", "periodic"]``.

    See Also
    --------
    :func:`morse_wavelet`, `morse_frequency`
    """
    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(x.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )
    # Positions and time arrays must have the same shape.
    if x.shape[time_axis] != wavelet.shape[-1]:
        raise ValueError("x and wave time axes must have the same length.")

    wavelet_ = np.moveaxis(wavelet, [freq_axis, order_axis], [-2, -3])

    # if x is of dimension 1 we need to expand
    # otherwise make sure time axis is last
    if np.ndim(x) < 2:
        x_ = np.expand_dims(x, axis=0)
    else:
        x_ = np.moveaxis(x, time_axis, -1)

    if ~np.all(np.isreal(x)):
        if normalization == "energy":
            x_ /= np.sqrt(2)
        elif normalization == "bandpass":
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

    time_length = np.shape(x)[time_axis]
    time_length_ = np.shape(x_)[-1]

    # pad wavelet with zeros: JML ok
    order_length, freq_length, _ = np.shape(wavelet)
    _wavelet = np.zeros((order_length, freq_length, time_length_), dtype=np.cdouble)

    index = slice(
        int(np.floor(time_length_ - time_length) / 2),
        int(time_length + np.floor(time_length_ - time_length) / 2),
    )
    _wavelet[:, :, index] = wavelet_

    # take fft along axis = -1
    _wavelet_fft = np.fft.fft(_wavelet)
    om = 2 * np.pi * np.linspace(0, 1 - 1 / time_length_, time_length_)
    if time_length_ % 2 == 0:
        _wavelet_fft = (
            _wavelet_fft
            * np.exp(1j * -om * (time_length_ + 1) / 2)
            * np.sign(np.pi - om)
        )
    else:
        _wavelet_fft = _wavelet_fft * np.exp(1j * -om * (time_length_ + 1) / 2)

    # here we should be able to automate the tiling without assuming extra dimensions of wave
    X_ = np.tile(
        np.expand_dims(np.fft.fft(x_), (-3, -2)),
        (1, order_length, freq_length, 1),
    )

    # finally the transform; return precision of input `x``; central part only
    complex_dtype = np.cdouble if x.dtype == np.single else np.csingle
    wtx = np.fft.ifft(X_ * np.conj(_wavelet_fft)).astype(complex_dtype)
    wtx = wtx[..., index]
    # remove extra dimensions
    wtx = np.squeeze(wtx)
    # reposition the time axis: should I add a condition to do so only if time_axis!=-1? works anyway
    wtx = np.moveaxis(wtx, -1, time_axis)

    return wtx


def morse_wavelet(
    length: int,
    gamma: float,
    beta: float,
    radian_frequency: np.ndarray,
    order: Optional[int] = 1,
    normalization: Optional[str] = "bandpass",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Morse wavelets of Olhede and Walden (2002).

    Parameters
    ----------
    length: int
       Length of the wavelet.
    gamma: float
       Gamma parameter of the wavelet.
    beta: float
       Beta parameter of the wavelet.
    radian_frequency: np.ndarray
       The radian frequencies at which the Fourier transform of the wavelets
       reach their maximum amplitudes. radian_frequency is between 0 and 2 * np.pi * 0.5,
       the normalized Nyquist radian frequency.
    order: int
        Wavelet order, default is 1.
    normalization: str, optional
       Normalization for the wavelets. Default is "bandpass".
       "bandpass" uses "bandpass normalization", meaning that the FFT of the wavelet
       has a peak value of 2 for all frequencies radian_frequency. "energy" uses the unit
       energy normalization. The time-domain wavelet energy np.sum(np.abs(wave)**2)
       is then always unity.

    Returns
    -------
    wave : np.ndarray
        Time-domain wavelets. wave will be of shape (length,np.size(radian_frequency),k).
    wavefft: np.ndarray
        Frequency-domain wavelets. wavefft will be of shape (length,np.size(radian_frequency),k).

    Examples
    --------
    To write.

    See Also
    --------
    :func:`wavelet_transform`, `morsefreq`
    """
    # add a test for radian_frequency being a numpy array
    # initialization
    wave = np.zeros((length, order, len(radian_frequency)), dtype=np.cdouble)
    wavefft = np.zeros((length, order, len(radian_frequency)), dtype=np.cdouble)

    # call to morse_wavelet take only gamma and be as float, no array
    fo, _, _ = morsefreq(gamma, beta)
    for i in range(len(radian_frequency)):
        wave_tmp = np.zeros((length, order), dtype=np.cdouble)
        wavefft_tmp = np.zeros((length, order), dtype=np.cdouble)

        # wavelet frequencies
        fact = np.abs(radian_frequency[i]) / fo
        # norm_radian_frequency first dim is n points
        norm_radian_frequency = (
            2 * np.pi * np.linspace(0, 1 - 1 / length, length) / fact
        )
        if normalization == "energy":
            if beta == 0:
                wavezero = np.exp(-(norm_radian_frequency**gamma))
            else:
                wavezero = np.exp(
                    beta * np.log(norm_radian_frequency)
                    - norm_radian_frequency**gamma
                )
        elif normalization == "bandpass":
            if beta == 0:
                wavezero = 2 * np.exp(-(norm_radian_frequency**gamma))
            else:
                wavezero = 2 * np.exp(
                    -beta * np.log(fo)
                    + fo**gamma
                    + beta * np.log(norm_radian_frequency)
                    - norm_radian_frequency**gamma
                )
        else:
            raise ValueError(
                "Normalization option (norm) must be one of 'energy' or 'bandpass'."
            )
        wavezero[0] = 0.5 * wavezero[0]
        # Replace NaN with zeros in wavezero
        wavezero = np.nan_to_num(wavezero, copy=False, nan=0.0)
        # second family is never used
        wavefft_tmp = _morse_wavelet_first_family(
            fact,
            gamma,
            beta,
            norm_radian_frequency,
            wavezero,
            order=order,
            normalization=normalization,
        )
        wavefft_tmp = np.nan_to_num(wavefft_tmp, posinf=0, neginf=0)
        # shape of wavefft_tmp is points, order
        # center wavelet
        norm_radian_frequency_mat = np.tile(
            np.expand_dims(norm_radian_frequency, -1), (order)
        )
        wavefft_tmp = wavefft_tmp * np.exp(
            1j * norm_radian_frequency_mat * (length + 1) / 2 * fact
        )
        # time domain wavelet
        wave_tmp = np.fft.ifft(wavefft_tmp, axis=0)
        if radian_frequency[i] < 0:
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


def _morse_wavelet_first_family(
    fact: float,
    gamma: float,
    beta: float,
    norm_radian_frequency: np.ndarray,
    wavezero: np.ndarray,
    order: Optional[int] = 1,
    normalization: Optional[str] = "bandpass",
) -> np.ndarray:
    """
    Derive first family wavelet. Internal use only.
    """
    r = (2 * beta + 1) / gamma
    c = r - 1
    L = np.zeros_like(norm_radian_frequency, dtype=np.float64)
    wavefft1 = np.zeros((np.shape(wavezero)[0], order))

    for i in np.arange(0, order):
        if normalization == "energy":
            A = morseafun(gamma, beta, order=i + 1, normalization=normalization)
            coeff = np.sqrt(1 / fact) * A
        elif normalization == "bandpass":
            if beta != 0:
                coeff = np.sqrt(np.exp(_lgamma(r) + _lgamma(i + 1) - _lgamma(i + r)))
            else:
                coeff = 1

        index = slice(
            0, int(np.round(np.shape(wavezero)[0] / 2))
        )  # how to define indices?
        L[index] = _laguerre(2 * norm_radian_frequency[index] ** gamma, i, c)
        wavefft1[:, i] = coeff * wavezero * L

    return wavefft1


def morsefreq(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
) -> Union[Tuple[np.ndarray], Tuple[float]]:
    """
    Frequency measures for generalized Morse wavelets. This functions calculates
    three different measures of the frequency of the lowest-order generalized Morse
    wavelet specified by parameters gamma (gamma) and beta (beta).

    Note that all frequency quantities here are *radian* as in cos(omega t)and not
    cyclic as in np.cos(2 np.pi f t).

    For be=0, the "wavelet" becomes an analytic lowpass filter, and fm
    is not defined in the usual way.  Instead, fm is defined as the point
    at which the filter has decayed to one-half of its peak power.

    For details see Lilly and Olhede (2009).  Higher-order properties of analytic
    wavelets.  IEEE Trans. Sig. Proc., 57 (1), 146--160.

    Parameters
    ----------
    gamma: np.ndarray or float
       Gamma parameter of the wavelet.
    beta: np.ndarray or float
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
    :func:`morse_wavelet`
    """
    # add test for type and shape in case of ndarray
    fm = np.where(
        beta == 0,
        np.log(2) ** (1 / gamma),
        np.exp((1 / gamma) * (np.log(beta) - np.log(gamma))),
    )

    fe = (
        1
        / (2 ** (1 / gamma))
        * _gamma((2 * beta + 2) / gamma)
        / _gamma((2 * beta + 1) / gamma)
    )

    fi = _gamma((beta + 2) / gamma) / _gamma((beta + 1) / gamma)

    return fm, fe, fi


def morsespace(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
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
    gamma: np.ndarray or float
       Gamma parameter of the wavelet.
    beta: np.ndarray or float
       Beta parameter of the wavelet.
    n:
    high:
    low:
    d:
    eta:

    Returns
    -------
    radian_frequency: np.ndarray

    Examples
    --------
    To write.

    """
    p, _, _ = morseprops(gamma, beta)

    _high = _morsehigh(gamma, beta, high)
    high_ = np.min(np.append(_high, np.pi))

    _low = 2 * np.sqrt(2) * p * low / n
    low_ = np.max(np.append(_low, 0))

    r = 1 + 1 / (d * p)
    m = np.floor(np.log10(high_ / low_) / np.log10(r))
    radian_frequency = high_ * np.ones(int(m + 1)) / r ** np.arange(0, m + 1)

    return radian_frequency


def _morsehigh(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
    eta: float,
) -> Union[np.ndarray, float]:
    """
    High-frequency cutoff of the generalized Morse wavelets.
    gamma and be should be of the same length.
    """
    m = 10000
    omhigh = np.linspace(0, np.pi, m)
    f = np.zeros_like(gamma, dtype="float")

    for i in range(0, len(gamma)):
        fm, _, _ = morsefreq(gamma[i], beta[i])
        om = fm * np.pi / omhigh
        lnwave1 = beta[i] / gamma[i] * np.log(np.exp(1) * gamma[i] / beta[i])
        lnwave2 = beta[i] * np.log(om) - om ** gamma[i]
        lnwave = lnwave1 + lnwave2
        index = np.nonzero(np.log(eta) - lnwave < 0)[0][0]
        f[i] = omhigh[index]

    return f


def morseprops(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
) -> Union[Tuple[np.ndarray], Tuple[float]]:
    """
    Properties of the demodulated generalized Morse wavelets.

    Parameters
    ----------
    gamma: np.ndarray or float
       Gamma parameter of the wavelet.
    beta: np.ndarray or float
       Beta parameter of the wavelet.

    Returns
    -------
    p: np.ndarray or float
    skew: np.ndarray or float
    kurt: np.ndarray or float
    """
    # test common size? or could be broadcasted
    p = np.sqrt(gamma * beta)
    skew = gamma - 3 / p
    kurt = 3 - skew**2 - 2 / p**2

    return p, skew, kurt


def morseafun(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
    order: Optional[np.int64] = 1,
    normalization: Optional[str] = "bandpass",
) -> float:
    # add test for type and shape in case of ndarray
    if normalization == "energy":
        r = (2 * beta + 1) / gamma
        a = (
            2
            * np.pi
            * gamma
            * (2**r)
            * np.exp(_lgamma(order) - _lgamma(order + r - 1))
        ) ** 0.5
    elif normalization == "bandpass":
        fm, _, _ = morsefreq(gamma, beta)
        a = np.where(beta == 0, 2, 2 / (np.exp(beta * np.log(fm) - fm**gamma)))
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

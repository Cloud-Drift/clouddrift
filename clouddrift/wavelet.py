"""
This module provides wavelet functions.
"""

import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import warnings
from math import gamma, lgamma


def morsewave(
    n: int,
    ga: float,
    be: float,
    fs: np.ndarray,
    k: Optional[int] = 1,
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
    k: int
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
    """
    # initialization
    psi = np.zeros((n, k, len(fs)), dtype="cdouble")
    psif = np.zeros((n, k, len(fs)), dtype="cdouble")

    # call to morsewave take only ga and be as float, no array
    fo, _, _ = morsefreq(ga, be)
    for i in range(0, len(fs)):
        psi_tmp = np.zeros((n, k), dtype="cdouble")
        psif_tmp = np.zeros((n, k), dtype="cdouble")

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
        # to do: replace NaN with zeros in psizero
        psizero = np.nan_to_num(psizero, copy=False, nan=0.0)
        # to do, derive second family wavelet, here do first family
        # spectral domain wavelet
        psif_tmp = _morsewave_first_family(fact, n, ga, be, om, psizero, k=k, norm=norm)
        psif_tmp = np.nan_to_num(psif_tmp, posinf=0, neginf=0)
        # shape of psif_tmp is points, order
        # center wavelet
        ommat = np.tile(np.expand_dims(om, -1), (k))
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

    # reorder dimension to follow jlab
    psi = np.swapaxes(psi, -2, -1)
    psif = np.swapaxes(psif, -2, -1)
    return psi, psif


def _morsewave_first_family(
    fact: float,
    n: int,
    ga: float,
    be: float,
    om: np.ndarray,
    psizero: np.ndarray,
    k: Optional[int] = 1,
    norm: Optional[str] = "bandpass",
) -> np.ndarray:
    """
    Derive first family wavelet
    """
    r = (2 * be + 1) / ga
    c = r - 1
    L = np.zeros_like(om, dtype="float")
    psif1 = np.zeros((np.shape(psizero)[0], k))

    for i in np.arange(0, k):
        if norm == "energy":
            A = morseafun(ga, be, k=i + 1, norm=norm)
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
    """
    # add test for type and shape in case of ndarray
    fm = np.where(
        be == 0, np.log(2) ** (1 / ga), np.exp((1 / ga) * (np.log(be) - np.log(ga)))
    )

    fe = 1 / (2 ** (1 / ga)) * _gamma((2 * be + 2) / ga) / _gamma((2 * be + 1) / ga)

    fi = _gamma((be + 2) / ga) / _gamma((be + 1) / ga)

    return fm, fe, fi


def morseafun(
    ga: Union[np.ndarray, float],
    be: Union[np.ndarray, float],
    k: Optional[np.int64] = 1,
    norm: Optional[str] = "bandpass",
) -> float:
    # add test for type and shape in case of ndarray
    if norm == "energy":
        r = (2 * be + 1) / ga
        a = (2 * np.pi * ga * (2**r) * np.exp(_lgamma(k) - _lgamma(k + r - 1))) ** 0.5
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

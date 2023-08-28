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
       reach their maximum amplitudes.
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
    psi = np.zeros((n, len(fs), k))
    psif = np.zeros((n, len(fs), k))

    # wavelet frequencies
    fo, _, _ = morsefreq(ga, be)
    fact = fs.flatten() / fo
    # om first dim is n points, second dim is frequencies
    om = 2 * np.pi * np.expand_dims(np.linspace(0, 1 - 1 / n, n), axis=-1) / fact

    if norm == "energy":
        psizero0 = np.exp(-(om**ga))
        psizero1 = np.exp(be * np.log(om) - om**ga)
        psizero = np.where(be == 0, psizero0, psizero1)
    elif norm == "bandpass":
        psizero0 = 2 * np.exp(-(om**ga))
        psizero1 = 2 * np.exp(-be * np.log(fo) + fo**ga + be * np.log(om) - om**ga)
        psizero = np.where(be == 0, psizero0, psizero1)
    else:
        raise ValueError(
            "Normalization option (norm) must be one of 'energy' or 'bandpass'."
        )

    # to do: psizero[0] = 0.5*psizero[0] but am not sure how
    # to do: replace NaN with zeros in psizero

    # to do, derive second family wavelet, here do first family
    # spectral domain wavelet
    psif = _morsewave_first_family(fact, n, ga, be, om, psizero, k=k, norm=norm)
    # center wavelet, see jlab
    # ommat=vrep(vrep(om,size(X,3),3),size(X,2),2);
    # Xr=X.*rot(ommat.*(N+1)/2*fact); %ensures wavelets are centered
    # time domain wavelet
    psi = np.fft.ifft(psif, axis=0)

    return psi, psif


def _morsewave_first_family(
    fact: np.ndarray,
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
    L = np.zeros_like(om)
    psif = np.zeros(
        (np.shape(psizero)[0], np.shape(psizero)[1], k)
    )  # len is like np.shape()[0]

    for i in np.arange(0, k):
        if norm == "energy":
            A = morseafun(k + 1, ga, be, norm=norm)
            coeff = np.sqrt(1 / fact) * A
        elif norm == "bandpass":
            if be == 0:
                coeff = 1
            else:
                coeff = np.sqrt(np.exp(_lgamma(r) + _lgamma(k + 1) - _lgamma(k + r)))

        index = slice(1, int(np.round(n / 2)))  # how to define indices?
        L[index, :] = _laguerre(2 * om[index, :] ** ga, k, c)
        psif[:, :, i] = coeff * psizero * L

    return psif


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
        a = (2 * np.pi * ga * (2**r) * np.exp(lgamma(k) - lgamma(k + r + 1))) ** 0.5
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
        y = np.zeros_like(x_)
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
        n = len(x)
        y = np.zeros_like(x)
        for i in range(0, n):
            y[i] = lgamma(x[i])
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
    y = np.zeros_like(x)
    for i in np.arange(0, k + 1):
        fact = np.exp(_lgamma(k + c + 1) - _lgamma(c + i + 1) - _lgamma(k - i + 1))
        y = y + (-1) ** i * fact * x**i / _gamma(i + 1)
    return y

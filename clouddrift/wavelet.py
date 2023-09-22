"""
This module provides functions for computing wavelet transforms and time-frequency analyses,
notably using generalized Morse wavelets.

The Python code in this module was translated from the MATLAB implementation
by J. M. Lilly in the jWavelet module of jLab (http://jmlilly.net/code.html).

Lilly, J. M. (2021), jLab: A data analysis package for Matlab, v.1.7.1,
doi:10.5281/zenodo.4547006, http://www.jmlilly.net/software.

jLab is licensed under the Creative Commons Attribution-Noncommercial-ShareAlike
License (https://creativecommons.org/licenses/by-nc-sa/4.0/). The code that is
directly translated from jLab/jWavelet is licensed under the same license.
Any other code that is added to this module and that is specific to Python and
not the MATLAB implementation is licensed under CloudDrift's MIT license.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy.special import gamma as _gamma, gammaln as _lgamma


def morse_wavelet_transform(
    x: np.ndarray,
    gamma: float,
    beta: float,
    radian_frequency: np.ndarray,
    complex: Optional[bool] = False,
    order: Optional[int] = 1,
    normalization: Optional[str] = "bandpass",
    boundary: Optional[str] = "mirror",
    time_axis: Optional[int] = -1,
) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    Apply a continuous wavelet transform to an input signal using the generalized Morse
    wavelets of Olhede and Walden (2002). The wavelet transform is normalized differently
    for complex-valued input than for real-valued input, and this in turns depends on whether the
    optional argument ``normalization`` is set to ``"bandpass"`` or ``"energy"`` normalizations.

    Parameters
    ----------
    x : np.ndarray
        Real- or complex-valued signals. The time axis is assumed to be the last. If not, specify optional
        argument `time_axis`.
    gamma : float
       Gamma parameter of the Morse wavelets.
    beta : float
       Beta parameter of the Morse wavelets.
    radian_frequency : np.ndarray
       An array of radian frequencies at which the Fourier transform of the wavelets
       reach their maximum amplitudes. ``radian_frequency`` is typically between 0 and 2 * np.pi * 0.5,
       the normalized Nyquist radian frequency.
    complex : boolean, optional
        Specify explicitely if the input signal ``x`` is a complex signal. Default is False which
        means that the input is real but that is not explicitely tested by the function.
        This choice affects the normalization of the outputs and their interpretation.
        See examples below.
    time_axis : int, optional
        Axis on which the time is defined for input ``x`` (default is last, or -1).
    normalization : str, optional
        Normalization for the wavelet transforms. By default it is assumed to be
        ``"bandpass"`` which uses a bandpass normalization, meaning that the FFT
        of the wavelets have peak value of 2 for all central frequencies
        ``radian_frequency``. However, if the optional argument ``complex=True``
        is specified, the wavelets will be divided by 2 so that the total
        variance of the input complex signal is equal to the sum of the
        variances of the returned analytic (positive) and anti-analiyic
        (negative) parts. See examples below. The other option is ``"energy"``
        which uses the unit energy normalization. In this last case, the
        time-domain wavelet energies ``np.sum(np.abs(wave)**2)`` are always
        unity.
    boundary : str, optional
        The boundary condition to be imposed at the edges of the input signal ``x``.
        Allowed values are ``"mirror"``, ``"zeros"``, and ``"periodic"``. Default is ``"mirror"``.
    order : int, optional
        Order of Morse wavelets, default is 1.

    Returns
    -------
    If the input signal is real as specificied by ``complex=False``:

    wtx : np.ndarray
        Time-domain wavelet transform of input ``x`` with shape ((x shape without time_axis), orders, frequencies, time_axis)
        but with dimensions of length 1 removed (squeezed).

    If the input signal is complex as specificied by ``complex=True``, a tuple is returned:

    wtx_p : np.array
        Time-domain positive wavelet transform of input ``x`` with shape ((x shape without time_axis), frequencies, orders),
        but with dimensions of length 1 removed (squeezed).
    wtx_n : np.array
        Time-domain negative wavelet transform of input ``x`` with shape ((x shape without time_axis), frequencies, orders),
        but with dimensions of length 1 removed (squeezed).

    Examples
    --------
    Apply a wavelet transform with a Morse wavelet with gamma parameter 3, beta parameter 4,
    at radian frequency 0.2 cycles per unit time:

    >>> x = np.random.random(1024)
    >>> wtx = morse_wavelet_transform(x, 3, 4, np.array([2*np.pi*0.2]))

    Apply a wavelet transform with a Morse wavelet with gamma parameter 3, beta parameter 4,
    for a complex input signal at radian frequency 0.2 cycles per unit time. This case returns the
    analytic and anti-analytic components:

    >>> z = np.random.random(1024) + 1j*np.random.random(1024)
    >>> wtz_p, wtz_n = morse_wavelet_transform(z, 3, 4, np.array([2*np.pi*0.2]), complex=True)

    The same result as above can be otained by applying the Morse transform on the real and imaginary
    component of z and recombining the results as follows for the "bandpass" normalization:
    >>> wtz_real = morse_wavelet_transform(np.real(z)), 3, 4, np.array([2*np.pi*0.2]))
    >>> wtz_imag = morse_wavelet_transform(np.imag(z)), 3, 4, np.array([2*np.pi*0.2]))
    >>> wtz_p, wtz_n = (wtz_real + 1j*wtz_imag) / 2, (wtz_real - 1j*wtz_imag) / 2

    For the "energy" normalization, the analytic and anti-analytic components are obtained as follows
    with this alternative method:
    >>> wtz_real = morse_wavelet_transform(np.real(z)), 3, 4, np.array([2*np.pi*0.2]))
    >>> wtz_imag = morse_wavelet_transform(np.imag(z)), 3, 4, np.array([2*np.pi*0.2]))
    >>> wtz_p, wtz_n = (wtz_real + 1j*wtz_imag) / np.sqrt(2), (wtz_real - 1j*wtz_imag) / np.sqrt(2)

    The input signal can have an arbitrary number of dimensions but its ``time_axis`` must be
    specified if it is not the last:

    >>> x = np.random.random((1024,10,15))
    >>> wtx = morse_wavelet_transform(x, 3, 4, np.array([2*np.pi*0.2]), time_axis=0)

    The default way to handle the boundary conditions is to mirror the ends points
    but this can be changed by specifying the chosen boundary method:

    >>> x = np.random.random((10,15,1024))
    >>> wtx = morse_wavelet_transform(x, 3, 4, np.array([2*np.pi*0.2]), boundary="periodic")

    This function can be used to conduct a time-frequency analysis of the input signal by specifying
    a range of randian frequencies using the ``morse_logspace_freq`` function as an example:

    >>> x = np.random.random(1024)
    >>> gamma = 3
    >>> beta = 4
    >>> radian_frequency = morse_logspace_freq(gamma, beta, np.shape(x)[0])
    >>> wtx = morse_wavelet_transform(x, gamma, beta, radian_frequency)


    Raises
    ------
    ValueError
        If the time axis is outside of the valid range ([-1, np.ndim(x)-1]).
        If boundary optional argument is not in ["mirror", "zeros", "periodic"]``.
        If normalization optional argument is not in ["bandpass", "energy"]``.

    See Also
    --------
    :func:`morse_wavelet`, `wavelet_transform`, `morse_logspace_freq`

    """
    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(x.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )
    # generate the wavelet
    wavelet, _ = morse_wavelet(
        np.shape(x)[time_axis],
        gamma,
        beta,
        radian_frequency,
        normalization=normalization,
        order=order,
    )

    # apply the wavelet transform, distinguish complex and real cases
    if complex:
        # imaginary case, divide by 2 the wavelet and return analytic and anti-analytic
        if normalization == "bandpass":
            wtx_p = wavelet_transform(
                0.5 * x, wavelet, boundary="mirror", time_axis=time_axis
            )
            wtx_n = wavelet_transform(
                np.conj(0.5 * x), wavelet, boundary="mirror", time_axis=time_axis
            )
        elif normalization == "energy":
            wtx_p = wavelet_transform(
                x / np.sqrt(2), wavelet, boundary="mirror", time_axis=time_axis
            )
            wtx_n = wavelet_transform(
                np.conj(x / np.sqrt(2)), wavelet, boundary="mirror", time_axis=time_axis
            )
        wtx = wtx_p, wtx_n

    elif ~complex:
        # real case
        wtx = wavelet_transform(x, wavelet, boundary=boundary, time_axis=time_axis)

    else:
        raise ValueError(
            "`complex` optional argument must be boolean 'True' or 'False'"
        )

    return wtx


def wavelet_transform(
    x: np.ndarray,
    wavelet: np.ndarray,
    boundary: Optional[str] = "mirror",
    time_axis: Optional[int] = -1,
    freq_axis: Optional[int] = -2,
    order_axis: Optional[int] = -3,
) -> np.ndarray:
    """
    Apply a continuous wavelet transform to an input signal using an input wavelet
    function. Such wavelet can be provided by the function ``morse_wavelet``.

    Parameters
    ----------
    x : np.ndarray
        Real- or complex-valued signals.
    wavelet : np.ndarray
        A suite of time-domain wavelets, typically returned by the function ``morse_wavelet``.
        The length of the time axis of the wavelets must be the last one and matches the
        length of the time axis of x. The other dimensions (axes) of the wavelets (such as orders and frequencies) are
        typically organized as orders, frequencies, and time, unless specified by optional arguments freq_axis and order_axis.
        The normalization of the wavelets is assumed to be "bandpass", if not, use kwarg normalization="energy", see ``morse_wavelet``.
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
        Time-domain wavelet transform of ``x`` with shape ((x shape without time_axis), orders, frequencies, time_axis)
        but with dimensions of length 1 removed (squeezed).

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

    Raises
    ------
    ValueError
        If the time axis is outside of the valid range ([-1, N-1]).
        If the shape of time axis is different for input signal and wavelet.
        If boundary optional argument is not in ["mirror", "zeros", "periodic"]``.

    See Also
    --------
    :func:`morse_wavelet`, `morse_wavelet_transform`, `morse_freq`
    """
    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(x.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )
    # Positions and time arrays must have the same shape.
    if x.shape[time_axis] != wavelet.shape[-1]:
        raise ValueError("x and wavelet time axes must have the same length.")

    wavelet_ = np.moveaxis(wavelet, [freq_axis, order_axis], [-2, -3])

    # if x is of dimension 1 we need to expand
    # otherwise make sure time axis is last
    if np.ndim(x) < 2:
        x_ = np.expand_dims(x, axis=0)
    else:
        x_ = np.moveaxis(x, time_axis, -1)

    # add detrending option eventually

    # apply boundary conditions
    if boundary == "mirror":
        x_ = np.concatenate((np.flip(x_, axis=-1), x_, np.flip(x_, axis=-1)), axis=-1)
    elif boundary == "zeros":
        x_ = np.concatenate((np.zeros_like(x_), x_, np.zeros_like(x_)), axis=-1)
    elif boundary == "periodic":
        pass
    else:
        raise ValueError("boundary must be one of 'mirror', 'zeros', or 'periodic'.")

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

    # reposition the time axis if needed from axis -1
    if time_axis != -1:
        wtx = np.moveaxis(wtx, -1, time_axis)

    # remove extra dimensions if needed
    wtx = np.squeeze(wtx)

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
    Compute the generalized Morse wavelets of Olhede and Walden (2002), doi: 10.1109/TSP.2002.804066.

    Parameters
    ----------
    length : int
       Length of the wavelets.
    gamma : float
       Gamma parameter of the wavelets.
    beta : float
       Beta parameter of the wavelets.
    radian_frequency : np.ndarray
       The radian frequencies at which the Fourier transform of the wavelets
       reach their maximum amplitudes. radian_frequency is between 0 and 2 * np.pi * 0.5,
       the normalized Nyquist radian frequency.
    order : int, optional
        Order of wavelets, default is 1.
    normalization : str, optional
       Normalization for the ``wavelet`` output. By default it is assumed to be ``"bandpass"``
       which uses a bandpass normalization, meaning that the FFT of the wavelets
       have peak value of 2 for all central frequencies ``radian_frequency``. The other option is
       ``"energy"``which uses the unit energy normalization. In this last case, the time-domain wavelet
       energies ``np.sum(np.abs(wave)**2)`` are always unity.

    Returns
    -------
    wavelet : np.ndarray
        Time-domain wavelets with shape (order, radian_frequency, length).
    wavelet_fft: np.ndarray
        Frequency-domain wavelets with shape (order, radian_frequency, length).

    Examples
    --------

    Compute a Morse wavelet with gamma parameter 3, beta parameter 4, at radian frequency 0.2 cycles per unit time:

    >>> wavelet, wavelet_fft = morse_wavelet(1024, 3, 4, np.array([2*np.pi*0.2]))
    >>> np.shape(wavelet)
    (1, 1, 1024)

    Compute a suite of Morse wavelets with gamma parameter 3, beta parameter 4, up to order 3,
    at radian frequencies 0.2 and 0.3 cycles per unit time:

    >>> wavelet, wavelet_fft = morse_wavelet(1024, 3, 4, np.array([2*np.pi*0.2, 2*np.pi*0.3]), order=3)
    >>> np.shape(wavelet)
    (3, 2, 1024)

    Compute a Morse wavelet specifying an energy normalization :
    >>> wavelet, wavelet_fft = morse_wavelet(1024, 3, 4, np.array([2*np.pi*0.2]), normalization="energy")

    Raises
    ------
    ValueError
        If normalization optional argument is not in ["bandpass", "energy"]``.

    See Also
    --------
    :func:`wavelet_transform`, `morse_wavelet_transform`, `morse_freq`, `morse_logspace_freq`, `morse_amplitude`, `morse_properties`
    """
    # ad test for radian_frequency being a numpy array
    # initialization
    wavelet = np.zeros((length, order, len(radian_frequency)), dtype=np.cdouble)
    waveletfft = np.zeros((length, order, len(radian_frequency)), dtype=np.cdouble)

    # call to morse_wavelet take only gamma and be as float, no array
    fo, _, _ = morse_freq(gamma, beta)
    for i in range(len(radian_frequency)):
        wavelet_tmp = np.zeros((length, order), dtype=np.cdouble)
        waveletfft_tmp = np.zeros((length, order), dtype=np.cdouble)

        # wavelet frequencies
        fact = np.abs(radian_frequency[i]) / fo
        # norm_radian_frequency first dim is n points
        norm_radian_frequency = (
            2 * np.pi * np.linspace(0, 1 - 1 / length, length) / fact
        )
        if normalization == "energy":
            with np.errstate(divide="ignore"):
                waveletzero = np.exp(
                    beta * np.log(norm_radian_frequency)
                    - norm_radian_frequency**gamma
                )
        elif normalization == "bandpass":
            if beta == 0:
                waveletzero = 2 * np.exp(-(norm_radian_frequency**gamma))
            else:
                with np.errstate(divide="ignore"):
                    waveletzero = 2 * np.exp(
                        -beta * np.log(fo)
                        + fo**gamma
                        + beta * np.log(norm_radian_frequency)
                        - norm_radian_frequency**gamma
                    )
        else:
            raise ValueError(
                "Normalization option (norm) must be one of 'energy' or 'bandpass'."
            )
        waveletzero[0] = 0.5 * waveletzero[0]
        # Replace NaN with zeros in waveletzero
        waveletzero = np.nan_to_num(waveletzero, copy=False, nan=0.0)
        # second family is never used
        waveletfft_tmp = _morse_wavelet_first_family(
            fact,
            gamma,
            beta,
            norm_radian_frequency,
            waveletzero,
            order=order,
            normalization=normalization,
        )
        waveletfft_tmp = np.nan_to_num(waveletfft_tmp, posinf=0, neginf=0)
        # shape of waveletfft_tmp is points, order
        # center wavelet
        norm_radian_frequency_mat = np.tile(
            np.expand_dims(norm_radian_frequency, -1), (order)
        )
        waveletfft_tmp = waveletfft_tmp * np.exp(
            1j * norm_radian_frequency_mat * (length + 1) / 2 * fact
        )
        # time domain waveletlet
        wavelet_tmp = np.fft.ifft(waveletfft_tmp, axis=0)
        if radian_frequency[i] < 0:
            wavelet[:, :, i] = np.conj(wavelet_tmp)
            waveletfft_tmp[1:-1, :] = np.flip(waveletfft_tmp[1:-1, :], axis=0)
            waveletfft[:, :, i] = waveletfft_tmp
        else:
            waveletfft[:, :, i] = waveletfft_tmp
            wavelet[:, :, i] = wavelet_tmp

    # reorder dimension to be (order, frequency, time steps)
    # enforce length 1 for first axis if order=1 (no squeezing)
    wavelet = np.moveaxis(wavelet, [0, 1, 2], [2, 0, 1])
    waveletfft = np.moveaxis(waveletfft, [0, 1, 2], [2, 0, 1])

    return wavelet, waveletfft


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
    Derive first family of Morse wavelets. Internal use only.
    """
    r = (2 * beta + 1) / gamma
    c = r - 1
    L = np.zeros_like(norm_radian_frequency, dtype=np.float64)
    wavefft1 = np.zeros((np.shape(wavezero)[0], order))

    for i in np.arange(0, order):
        if normalization == "energy":
            A = morse_amplitude(gamma, beta, order=i + 1, normalization=normalization)
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


def morse_freq(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
) -> Union[Tuple[np.ndarray], Tuple[float]]:
    """
    Frequency measures for generalized Morse wavelets. This functions calculates
    three different measures fm, fe, and fi of the frequency of the lowest-order generalized Morse
    wavelet specified by parameters ``gamma`` and ``beta``.

    Note that all frequency quantities here are in *radian* as in cos(f t) and not
    cyclic as in np.cos(2 np.pi f t).

    For ``beta=0``, the corresponding wavelet becomes an analytic lowpass filter, and fm
    is not defined in the usual way but as the point at which the filter has decayed
    to one-half of its peak power.

    For details see Lilly and Olhede (2009), doi: 10.1109/TSP.2008.2007607.

    Parameters
    ----------
    gamma : np.ndarray or float
       Gamma parameter of the wavelets.
    beta : np.ndarray or float
       Beta parameter of the wavelets.

    Returns
    -------
    fm : np.ndarray
        The modal or peak frequency.
    fe : np.ndarray
        The energy frequency.
    fi : np.ndarray
        The instantaneous frequency at the wavelets' centers.

    Examples
    --------

    >>> fm, fe, fi = morse_freq(3,4)

    >>> morse_freq(3,4)
    (array(1.10064242), 1.1025129235952809, 1.1077321674324723)

    >>> morse_freq(3,np.array([10,20,30]))
    (array([1.49380158, 1.88207206, 2.15443469]),
    array([1.49421505, 1.88220264, 2.15450116]),
    array([1.49543843, 1.88259299, 2.15470024]))

    >>> morse_freq(np.array([3,4,5]),np.array([10,20,30]))
    (array([1.49380158, 1.49534878, 1.43096908]),
    array([1.49421505, 1.49080278, 1.4262489 ]),
    array([1.49543843, 1.48652036, 1.42163583]))

    >>> morse_freq(np.array([3,4,5]),10)
    (array([1.49380158, 1.25743343, 1.14869835]),
    array([1.49421505, 1.25000964, 1.13759731]),
    array([1.49543843, 1.24350315, 1.12739747]))

    See Also
    --------
    :func:`morse_wavelet`, `morse_amplitude`
    """
    # add test for type and shape in case of ndarray?
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


def morse_logspace_freq(
    gamma: float,
    beta: float,
    length: int,
    highset: Optional[Tuple[float]] = (0.1, np.pi),
    lowset: Optional[Tuple[float]] = (5, 0),
    density: Optional[int] = 4,
) -> np.ndarray:
    """
    Compute logarithmically-spaced frequencies for generalized Morse wavelets
    with parameters gamma and beta. This is a useful function to obtain the frequencies
    needed for time-frequency analyses using wavelets. If ``radian_frequencies`` is the
    output, ``np.log(radian_frequencies)`` is uniformly spaced, following convention
    for wavelet analysis. See Lilly (2017), doi: 10.1098/rspa.2016.0776.

    Default settings to compute the frequencies can be changed by passing optional
    arguments ``lowset``, ``highset``, and ``density``. See below.

    Parameters
    ----------
    gamma : float
       Gamma parameter of the Morse wavelets.
    beta : float
       Beta parameter of the Morse wavelets.
    length : int
        Length of the Morse wavelets and input signals.
    highset : tuple of floats, optional.
        Tuple of values (eta, high) used for high-frequency cutoff calculation. The highest
        frequency is set to be the minimum of a specified value and a cutoff frequency
        based on a Nyquist overlap condition: the highest frequency is the minimum of
        the specified value high, and the largest frequency for which the wavelet will
        satisfy the threshold level eta. Here eta be a number between zero and one
        specifying the ratio of a frequency-domain wavelet at the Nyquist frequency
        to its peak value. Default is (eta, high) = (0.1, np.pi).
    lowset : tuple of floats, optional.
        Tupe of values (P, low) set used for low-frequency cutoff calculation based on an
        endpoint overlap condition. The lowest frequency is set such that the lowest-frequency
        wavelet will reach some number P, called the packing number, times its central window
        width at the ends of the time series. A choice of P=1 corresponds to  roughly 95% of
        the time-domain wavelet energy being contained within the time series endpoints for
        a wavelet at the center of the domain. The second value of the tuple is the absolute
        lowest frequency. Default is (P, low) = (5, 0).
    density : int, optional
        This optional argument controls the number of points in the returned frequency
        array. Higher values of ``density`` mean more overlap in the frequency
        domain between transforms. When ``density=1``, the peak of one wavelet is located at the
        half-power points of the adjacent wavelet. The default ``density=4`` means
        that four other wavelets will occur between the peak of one wavelet and
        its half-power point.

    Returns
    -------
    radian_frequency : np.ndarray
        Logarithmically-spaced frequencies in radians cycles per unit time,
        sorted in descending order.

    Examples
    --------
    Generate a frequency array for the generalized Morse wavelet
    with parameters gamma=3 and beta=5 for a time series of length n=1024:

    >>> radian_frequency = morse_logspace_freq(3,5,1024)

    >>> radian_frequency = morse_logspace_freq(3,5,1024,highset=(0.2,np.pi),lowset=(5,0))

    >>> radian_frequency = morse_logspace_freq(3,5,1024,highset=(0.2,np.pi),lowset=(5,0),density=10)

    See Also
    --------
    :func:`morse_wavelet`, `morse_freq`, `morse_properties`.
    """
    gamma_ = np.array([gamma])
    beta_ = np.array([beta])
    width, _, _ = morse_properties(gamma_, beta_)

    _high = _morsehigh(gamma_, beta_, highset[0])
    high_ = np.min(np.append(_high, highset[1]))

    low = 2 * np.sqrt(2) * width * lowset[0] / length
    low_ = np.max(np.append(low, lowset[1]))

    r = 1 + 1 / (density * width)
    m = np.floor(np.log10(high_ / low_) / np.log10(r))
    radian_frequency = high_ * np.ones(int(m + 1)) / r ** np.arange(0, m + 1)

    return radian_frequency


def _morsehigh(
    gamma: np.ndarray,
    beta: np.ndarray,
    eta: float,
) -> Union[np.ndarray, float]:
    """High-frequency cutoff of the generalized Morse wavelets.
    gamma and be should be arrays of the same length. Internal use only.
    """
    m = 10000
    omhigh = np.linspace(0, np.pi, m)
    f = np.zeros_like(gamma, dtype="float")

    for i in range(0, len(gamma)):
        fm, _, _ = morse_freq(gamma[i], beta[i])
        with np.errstate(all="ignore"):
            om = fm * np.pi / omhigh
            lnwave1 = beta[i] / gamma[i] * np.log(np.exp(1) * gamma[i] / beta[i])
            lnwave2 = beta[i] * np.log(om) - om ** gamma[i]
            lnwave = lnwave1 + lnwave2
        index = np.nonzero(np.log(eta) - lnwave < 0)[0][0]
        f[i] = omhigh[index]

    return f


def morse_properties(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
) -> Union[Tuple[np.ndarray], Tuple[float]]:
    """
    Calculate the properties of the demodulated generalized Morse wavelets.
    See Lilly and Olhede (2009), doi: 10.1109/TSP.2008.2007607.

    Parameters
    ----------
    gamma : np.ndarray or float
       Gamma parameter of the wavelets.
    beta : np.ndarray or float
       Beta parameter of the wavelets.

    Returns
    -------
    width : np.ndarray or float
        Dimensionless time-domain window width of the wavelets.
    skew : np.ndarray or float
        Imaginary part of normalized third moment of the time-domain demodulate,
        or 'demodulate skewness'.
    kurt : np.ndarray or float
        Normalized fourth moment of the time-domain demodulate,
        or 'demodulate kurtosis'.

    See Also
    --------
    :func:`morse_wavelet`, `morse_freq`, `morse_amplitude`, `morse_logspace_freq`.
    """
    # test common size? or could be broadcasted
    width = np.sqrt(gamma * beta)
    skew = (gamma - 3) / width
    kurt = 3 - skew**2 - 2 / width**2

    return width, skew, kurt


def morse_amplitude(
    gamma: Union[np.ndarray, float],
    beta: Union[np.ndarray, float],
    order: Optional[np.int64] = 1,
    normalization: Optional[str] = "bandpass",
) -> float:
    """
    Calculate the amplitude coefficient of the generalized Morse wavelets.
    By default, the amplitude is calculated such that the maximum of the
    frequency-domain wavelet is equal to 2, which is the bandpass normalization.
    Optionally, specify ``normalization="energy"`` in order to return the coefficient
    giving the wavelets unit energies. See Lilly and Olhede (2009), doi doi: 10.1109/TSP.2008.2007607.

    Parameters
    ----------
    gamma : np.ndarray or float
       Gamma parameter of the wavelets.
    beta : np.ndarray or float
       Beta parameter of the wavelets.
    order : int, optional
        Order of wavelets, default is 1.
    normalization : str, optional
       Normalization for the wavelets. By default it is assumed to be ``"bandpass"``
       which uses a bandpass normalization, meaning that the FFT of the wavelets
       have peak value of 2 for all central frequencies ``radian_frequency``. The other option is ``"energy"``
       which uses the unit energy normalization. In this last case the time-domain wavelet
       energies ``np.sum(np.abs(wave)**2)`` are always unity.

    Returns
    -------
    amp : np.ndarray or float
        The amplitude coefficient of the wavelets.

    See Also
    --------
    :func:`morse_wavelet`, `morse_freq`, `morse_props`, `morse_logspace_freq`.
    """
    # add test for type and shape in case of ndarray
    if normalization == "energy":
        r = (2 * beta + 1) / gamma
        amp = (
            2
            * np.pi
            * gamma
            * (2**r)
            * np.exp(_lgamma(order) - _lgamma(order + r - 1))
        ) ** 0.5
    elif normalization == "bandpass":
        fm, _, _ = morse_freq(gamma, beta)
        amp = np.where(beta == 0, 2, 2 / (np.exp(beta * np.log(fm) - fm**gamma)))
    else:
        raise ValueError(
            "Normalization option (normalization) must be one of 'energy' or 'bandpass'."
        )

    return amp


def _laguerre(
    x: Union[np.ndarray, float],
    k: float,
    c: float,
) -> np.ndarray:
    """Generalized Laguerre polynomials"""
    y = np.zeros_like(x, dtype="float")
    for i in np.arange(0, k + 1):
        fact = np.exp(_lgamma(k + c + 1) - _lgamma(c + i + 1) - _lgamma(k - i + 1))
        y = y + (-1) ** i * fact * x**i / _gamma(i + 1)
    return y

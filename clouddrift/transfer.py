"""
This module provides functions to calculate various transfer function from wind stress to oceanic
velocity.
"""

import warnings
from typing import Callable, Optional

import numpy as np
from numpy import floating as _Floating
from numpy.lib.scimath import sqrt
from scipy.special import factorial, iv, kv  # type: ignore

from clouddrift.sphere import EARTH_DAY_SECONDS


def apply_transfer_function(
    x: np.ndarray,
    transfer_func: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    dt: Optional[float] = None,
) -> np.ndarray:
    """
    Applies a transfer function in the frequency domain to an input time series.

    Parameters
    ----------
        x:  array_like
            Input one-dimensional time series.
        transfer_func: callable or numpy ndarray
            Transfer function to apply to the input `x`.
                - If a callable, it should accept a frequency array as input as shown in the examples below.
                - If a numpy ndarray, it must match the length of input `x` and represents the transfer function
            in the frequency domain, where the first half corresponds to positive frequencies and the
            second half corresponds to negative frequencies.
        dt: float, optional
            Time step size of the input time series, needed if `transfer_func` is callable, otherwise ignored.

    Returns
    -------
        y: numpy ndarray
            Output time series after applying the transfer function.

    Examples
    --------
        Apply a low-pass filter transfer function to a random time series:
        >>> import numpy as np
        >>> from clouddrift.transfer import apply_transfer_function
        >>> x = np.random.randn(100)
        >>> transfer_func = lambda omega: 1 / (1 + (omega / (2 * np.pi * 0.1))**2)  # Low-pass filter
        >>> dt = 1.0  # Time step needed for callable transfer functions
        >>> y = apply_transfer_function(x, transfer_func, dt)

        Directly provide as an array a transfer function which suppresses
        the negative frequencies of a complex time series:
        >>> import numpy as np
        >>> from clouddrift.transfer import apply_transfer_function
        >>> x = np.random.randn(100) + 1j * np.random.randn(100)
        >>> transfer_func = np.ones(100)
        >>> transfer_func[50:] = 0  # Suppress negative frequencies
        >>> y = apply_transfer_function(x, transfer_func)

        To a random wind stress, apply a slab wind transfer function at latitude 45 degrees,
        with a friction coefficient of 0.1 times the Coriolis frequency and a boundary layer
        depth of 50 meters:
        >>> import numpy as np
        >>> from clouddrift.sphere import coriolis_frequency, EARTH_DAY_SECONDS
        >>> from clouddrift.transfer import apply_transfer_function, slab_wind_transfer
        >>> tau = np.random.randn(100) + 1j * np.random.randn(100)  # Random wind stress
        >>> cor_freq = coriolis_frequency(45) * EARTH_DAY_SECONDS  # Coriolis frequency at 45 degrees latitude in radians per day
        >>> transfer_slab = lambda omega: slab_wind_transfer(omega, cor_freq[0] , friction=0.1 * cor_freq[0] / EARTH_DAY_SECONDS, bld=50)
        >>> dt = 1.0  # Time step needed for callable transfer functions
        >>> u = apply_transfer_function(tau, transfer_slab, dt)

    Raises
    ------
        ValueError
            If input `x` is not a one-dimensional array.
            If `transfer_func` is callable but `dt` is not provided.
            If the shape of `transfer_func` does not match the shape of `x`.
    """

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("Input x must be a 1D array.")
    n = x.size

    Xf = np.fft.fft(x)

    if callable(transfer_func):
        if dt is None:
            raise ValueError(
                "The `dt` argument must be provided when `transfer_func` is callable."
            )
        omega = 2 * np.pi * np.fft.fftfreq(n, dt)
        G = transfer_func(omega)
    else:
        if dt is not None:
            warnings.warn(
                "The `dt` argument is ignored when `transfer_func` is provided as an array.",
                UserWarning,
                stacklevel=2,
            )
        G = np.asarray(transfer_func)
        if G.shape != Xf.shape:
            raise ValueError(
                "Transfer function array must match the shape of input `x`."
            )

    Yf = G * Xf
    y = np.fft.ifft(Yf)

    return y


def _epanechnikov_kernel(window_size):
    # Epanechnikov kernel: k(x) = 0.75 * (1 - x^2) for |x| <= 1
    x = np.linspace(-1, 1, window_size + 2)[
        1:-1
    ]  # Exclude endpoints to avoid null weights
    kernel = 0.75 * (1 - x**2)
    return kernel


def apply_sliding_transfer_function(
    x: np.ndarray,
    transfer_func: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    window_size: Optional[int] = None,
    step: int = 1,
    dt: Optional[float] = None,
    kernel: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies a transfer function in the frequency domain to a one-dimensional input time series in a
    sliding window fashion.
    This function returns the result for each window, as well as a weighted average of the results
    across all windows using a uniform (default) or Epanechnikov kernel.
    Mirror boundary conditions are used to extend the input time series so that the sliding window
    can be applied without losing data at the edges.

    Parameters
    ----------
        x:  array_like
            Input one-dimensional time series.
        transfer_func: callable or numpy ndarray
            Transfer function to apply to the input `x`.
                - If a callable, it should accept a frequency array as input as shown in the examples below.
                - If a numpy ndarray, it must match the length of input `x` and represents the transfer function
            in the frequency domain, where the first half corresponds to positive frequencies and the
            second half corresponds to negative frequencies.
        window_size: int, optional
            Size of the sliding window, needed if `transfer_func` is callable. It cannot be larger
            than the length of `x`.
        step: int
            Step size for the sliding window. Default is 1. Consecutive windows will not overlap if
            `step` is equal to or larger than `window_size` or the length of `transfer_func` if this
            one is a numpy array.
        dt: float, optional
            Time step size of the input time series, needed if `transfer_func` is callable, otherwise ignored.
        kernel: str, optional
            Type of kernel to use for weighting. Options are "uniform" (default), "epanechnikov",
            or any first letter of either. For non-overlapping windows, the chosen kernel is  irrelevant.

    Returns
    -------
        result: tuple of np.ndarray
            Tuple containing:
                - Array of results for each window (shape: (num_windows, window_size))
                - Weighted average result (shape: (n,))

    Examples
    --------
        Apply a low-pass filter in a sliding window:
        >>> import numpy as np
        >>> from clouddrift.transfer import apply_sliding_transfer_function
        >>> x = np.random.rand(100)
        >>> window_size = 10
        >>> step = 1
        >>> dt = 1.0
        >>> transfer_func = lambda omega: 1 / (1 + (omega / (2 * np.pi * 0.1))**2)  # Low-pass filter
        >>> result = apply_sliding_transfer_function(x, transfer_func, window_size, step, dt)
        The shape of the result is here (len(`x`) // `step` + 1, `window_size`) = (101, 10).

        Apply a downscaling function in non-overlapping windows:
        >>> import numpy as np
        >>> from clouddrift.transfer import apply_sliding_transfer_function
        >>> x = np.random.rand(100)
        >>> window_size = 10
        >>> step = 11
        >>> transfer_func = np.ones(window_size) / np.sqrt(2)
        >>> results, average_result = apply_sliding_transfer_function(x, transfer_func, window_size, step)
        The shape of results is here (len(`x`) // `step` + 1, `window_size`) = (91, 10).

        To a random wind stress hourly time series, apply an Ekman transfer function in a sliding window
        for an infinite boundary layer depth, an Ekman depth of 10 m, in order to obtain the Ekman
        velocity at a depth of 15 m, and use an Epanechnikov kernel for weighting:
        >>> import numpy as np
        >>> from clouddrift.sphere import coriolis_frequency, EARTH_DAY_SECONDS
        >>> from clouddrift.transfer import apply_transfer_function, slab_wind_transfer
        >>> tau = np.random.randn(240) + 1j * np.random.randn(240)  # Random wind stress
        >>> cor_freq = coriolis_frequency(45) * EARTH_DAY_SECONDS  # Coriolis frequency at 45 degrees latitude in radians per day
        >>> transfer_ekman = lambda omega: wind_transfer(omega, z=15, cor_freq= cor_freq,
        delta=10.0, mu=0.0, bld=np.inf, boundary_condition="no-slip")[0].squeeze()
        >>> dt = 1 / 24  # Time step needed for callable transfer functions
        >>> results, average_result = apply_sliding_transfer_function(tau, transfer_ekman, window_size=24, step=12, dt=dt, kernel="epanechnikov")
        The shape of results is here (len(`tau`) // `step` + 1, `window_size`) = (21, 24).

    Raises
    ------
        ValueError
            If `transfer_func` is a numpy array which is not 1D.
            If `transfer_func` is a numpy array which length does not match `window_size` if this one is provided.
            If `window_size` or `step` is not a positive integer.
            If `window_size` is greater than the length of `x`.
            If `step` is not a positive integer.
            If `dt` is not provided when `transfer_func` is callable.
            If `kernel` is not one of the allowed strings ("uniform", "epanechnikov", or their first letters).
        Warning
            If `dt` is provided when `transfer_func` is a numpy array, as it is ignored in this case.
    """
    x = np.asarray(x)
    n = x.size

    if callable(transfer_func):
        if window_size is None:
            raise ValueError(
                "`window_size` must be provided when transfer_func is callable."
            )
        if not isinstance(window_size, int) or window_size <= 0 or window_size > n:
            raise ValueError(
                "`window_size` must be a positive integer no greater than the length of `x`."
            )
        if not isinstance(step, int) or step <= 0:
            raise ValueError("`step` must be a positive integer.")
        if dt is None:
            raise ValueError(
                "`dt` (time step of input) must be provided when `transfer_func` is callable."
            )

    if not callable(transfer_func) and isinstance(transfer_func, np.ndarray):
        if transfer_func.ndim != 1:
            raise ValueError("`transfer_func` must be a 1D numpy array.")
        if window_size is None:
            window_size = transfer_func.shape[0]
        if window_size != transfer_func.shape[0]:
            raise ValueError(
                "When `transfer_func` is a numpy ndarray, its length must match `window_size` if this one"
                " is provided."
            )
        if not isinstance(step, int) or step <= 0:
            raise ValueError("`step` must be a positive integer.")
        if dt is not None:
            warnings.warn(
                "The `dt` argument is ignored when `transfer_func` is provided as an array.",
                stacklevel=2,
            )

    # Kernel selection (default is uniform)
    kernel_str = str(kernel).lower() if kernel is not None else "uniform"
    if window_size is None:
        raise ValueError("window_size must be specified and not None.")
    if kernel_str.startswith("e"):
        kernel_arr = _epanechnikov_kernel(window_size)
    elif kernel_str.startswith("u") or kernel_str == "":
        kernel_arr = np.ones(window_size)
    else:
        raise ValueError(
            "kernel must be 'uniform', 'epanechnikov', or their first letter."
        )

    # Extend x by half window_size at the beginning and end using mirror boundary condition
    before_window = window_size // 2
    after_window = window_size - before_window
    before_mirror = (
        x[1 : before_window + 1][::-1] if before_window > 0 else np.array([])
    )
    after_mirror = x[-after_window - 1 : -1][::-1] if after_window > 0 else np.array([])
    x_ = np.concatenate([before_mirror, x, after_mirror])

    n_ = (
        x_.size
    )  # length of extended array is n + before_window + after_window = n + window_size

    # initialize arrays for results
    result_sum = np.zeros(n_, dtype=np.complex128)
    result_weight = np.zeros(n_, dtype=np.float64)
    # result = []
    if window_size is None:
        raise ValueError("`window_size` must be specified and not None.")
    # number of windows is (n_ - window_size) // step + 1 = n // step + 1
    result = np.zeros((n // step + 1, window_size), dtype=np.complex128)

    # Sliding window application
    for start in range(0, n_ - window_size + 1, step):
        x_tmp = x_[start : start + window_size]
        if callable(transfer_func):
            result_tmp = apply_transfer_function(x_tmp, transfer_func, dt)
            result[start // step, :] = result_tmp
        else:
            if transfer_func.ndim == 1:
                result_tmp = apply_transfer_function(x_tmp, transfer_func)
                result[start // step, :] = result_tmp
            else:
                raise ValueError(
                    "`transfer_func` must be callable or a 1D numpy array."
                )
        weighted_result_tmp = result_tmp * kernel_arr
        result_sum[start : start + window_size] += weighted_result_tmp
        result_weight[start : start + window_size] += kernel_arr

    # average the results
    average_result = result_sum / result_weight
    # truncate the average_result to the original length of x
    average_result = average_result[before_window : before_window + n]

    return result, average_result


def slab_wind_transfer(
    omega: float | np.ndarray,
    cor_freq: float,
    friction: float,
    bld: float,
    density: float = 1025.0,
) -> np.ndarray:
    """
    Compute the the transfer function in the case of a ocean slab layer.

    Parameters
    ----------
        omega: array_like or float
            Angular frequency of the wind stress forcing in radians per day.
        cor_freq: float
            Coriolis frequency, in radians per day.
        friction: float
            Friction coefficient, in s-1.
        bld: float
            Thickness of the slab layer, in meters.
        density: float
            Seawater density, in kg m-3. Default is 1025.
    """
    # check that the boundary layer depth is positive
    if bld < 0:
        raise ValueError("Boundary layer depth bld must be positive.")

    # check that the density is positive
    if density < 0:
        raise ValueError("Density density must be positive.")

    # omega can be scalars or arrays, convert to arrays here
    omega_ = np.atleast_1d(omega)

    # convert to radians per second
    omega_ = omega_ / EARTH_DAY_SECONDS
    cor_freq_ = cor_freq / EARTH_DAY_SECONDS

    G = 1 / (density * bld * (friction + 1j * (omega_ + cor_freq_)))

    return G


def wind_transfer(
    omega: float | np.ndarray,
    z: float | np.ndarray,
    cor_freq: float,
    delta: float,
    mu: float,
    bld: float,
    boundary_condition: str = "no-slip",
    method: str = "lilly",
    density: float = 1025.0,
) -> tuple[
    float | _Floating | np.ndarray,
    float | _Floating | np.ndarray,
    float | _Floating | np.ndarray,
]:
    """
    Compute the transfer function from wind stress to oceanic velocity based on the physically-based
    models of Elipot and Gille (2009) and Lilly and Elipot (2021).

    Parameters
    ----------
        omega: array_like or float
            Angular frequency of the wind stress forcing in radians per day.
        z: array_like or float
            Depth in the ocean, positive downwards, in meters.
        cor_freq: float
            Coriolis frequency, in radians per day.
        delta: float
            Ekman depth, in meters.
        mu: float
            Madsen depth, in meters.
        bld: float
            Boundary layer depth, in meters.
        boundary_condition: str
            Bottom boundary condition at the base of the ocean surface boundary layer.
            Options are "no-slip" (Default) or "free-slip".
        method: str
            Method to compute the transfer function. Options are "lilly" (Default and preferred method)
            or "elipot".
        density: float
            Seawater density, in kg m-3. Default is 1025.

    Returns
    -------
        G: np.ndarray
            The transfer function from wind stress to oceanic velocity in units of kg-1 m 2 s.
        ddelta: np.ndarray
            The gradient of transfer function from the rate of change of the Ekman depth.
        dh: np.ndarray
            The gradient of transfer function from to the rate of change of the boundary layer depth.

    Examples
    --------
        To calculate the transfer function of Lilly and Elipot (2021) corresponding to the Ekman case
         of infinite depth ocean surface boundary layer and constant vertical eddy viscosity, at the zero frequency:

         >>> omega = 0
         >>> z = np.linspace(0, 100, 100)
         >>> cor_freq = 2 * np.pi * 1
         >>> K0 = 1e-4
         >>> delta = np.sqrt(2 * K0 / cor_freq / 86400)
         >>> mu = 0
         >>> G = wind_transfer(omega, z, cor_freq, delta, mu, np.inf, "no-slip")

    Raises
    ------
    ValueError
        If the depth z is not positive.

    References
    ----------
    [1] Elipot, S., and S. T. Gille (2009). Ekman layers in the Southern Ocean: spectral models and
    observations, vertical viscosity and boundary layer depth Ocean Sci., 5, 115–139, 2009, doi:10.5194/os-5-115-2009.

    [2] Lilly, J. M. and S. Elipot (2021). A unifying perspective on transfer function solutions to
    the unsteady Ekman problem. Fluids, 6 (2): 85, 1--36. doi:10.3390/fluids6020085.

    """
    # check that z is positive
    if np.any(z < 0):
        raise ValueError("Depth z must be positive.")

    # check that the boundary layer depth is positive
    if bld < 0:
        raise ValueError("Boundary layer depth bld must be positive.")

    # check that the Ekman depth is positive
    if delta < 0:
        raise ValueError("Ekman depth delta must be positive.")

    # check that the Madsen depth is positive
    if mu < 0:
        raise ValueError("Madsen depth mu must be positive.")

    # check that the density is positive
    if density < 0:
        raise ValueError("Density density must be positive.")

    # check that the boundary condition is valid
    if boundary_condition not in ["no-slip", "free-slip"]:
        raise ValueError("Boundary condition must be 'no-slip' or 'free-slip'.")

    # check that the method is valid
    if method not in ["lilly", "elipot"]:
        raise ValueError("Method must be 'lilly' or 'elipot'.")

    # z and omega can be scalars or arrays, convert to arrays here
    z_ = np.atleast_1d(z)
    omega_ = np.atleast_1d(omega)

    # convert to radians per second
    omega_ = omega_ / EARTH_DAY_SECONDS
    cor_freq_ = cor_freq / EARTH_DAY_SECONDS

    # Create the gridmesh of frequency and depth
    [omega_grid, z_grid] = np.meshgrid(omega_, z_)

    if boundary_condition == "no-slip":
        if method == "lilly":
            G = _wind_transfer_no_slip(
                omega_grid,
                z_grid,
                cor_freq_,
                delta,
                mu,
                bld,
                density,
            )
        else:
            G = _wind_transfer_elipot_no_slip(
                omega_grid,
                z_grid,
                cor_freq_,
                delta,
                mu,
                bld,
                density,
            )
    elif boundary_condition == "free-slip":
        if method == "lilly":
            G = _wind_transfer_free_slip(
                omega_grid,
                z_grid,
                cor_freq_,
                delta,
                mu,
                bld,
                density,
            )
        else:
            G = _wind_transfer_elipot_free_slip(
                omega_grid,
                z_grid,
                cor_freq_,
                delta,
                mu,
                bld,
                density,
            )

    # set G to nan where z > bld; may be mathematcially valid but not physically meaningful
    G[z_grid > bld] = np.nan

    # analytical gradients of the transfer function for mu = 0 and free slip, lilly method
    if boundary_condition == "no-slip" and method == "lilly" and mu == 0:
        s = np.sign(cor_freq_) * np.sign(1 + omega_grid / cor_freq_)
        Gamma = sqrt(2) * _rot(s * np.pi / 4) * sqrt(np.abs(1 + omega_grid / cor_freq_))
        ddelta1 = (
            (Gamma * (bld / delta) * np.tanh(Gamma * (bld / delta)) - 1) * G / delta
        )

        numer = np.exp(Gamma * (-z_grid / delta)) + np.exp(
            -Gamma * (2 * bld - z_grid) / delta
        )
        denom = 1 + np.exp(-Gamma * (2 * bld / delta))
        ddelta2 = (
            -2
            / (delta**2 * np.abs(cor_freq_) * density)
            * (bld - z_grid)
            / delta
            * (numer / denom)
        )

        dG_ddelta = ddelta1 + ddelta2

        dbld1 = -Gamma / delta * np.tanh(Gamma * (bld / delta)) * G
        dbld2 = 2 / (delta**2 * np.abs(cor_freq_) * density) * (numer / denom)
        dG_dbld = dbld1 + dbld2

    else:
        dG_ddelta = np.array([], dtype=float)  # Empty array for consistency
        dG_dbld = np.array([], dtype=float)

    return G, dG_ddelta, dG_dbld


def _wind_transfer_no_slip(
    omega_grid: np.ndarray,
    z_grid: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Transfer function from wind stress to oceanic velocity with no-slip boundary condition.
    """

    zo = np.divide(delta**2, mu)
    s = np.sign(coriolis_frequency) * np.sign(1 + omega_grid / coriolis_frequency)

    xiz, xih, xi0 = _xis(s, zo, delta, z_grid, omega_grid, coriolis_frequency, bld)

    if bld == np.inf:
        if mu == 0:
            # Ekman solution
            coeff = (sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta * np.abs(coriolis_frequency) * density
            )
            G = (
                coeff
                * (
                    np.exp(
                        -(1 + s * 1j)
                        * (z_grid / delta)
                        * sqrt(np.abs(1 + omega_grid / coriolis_frequency))
                    )
                )
                / (sqrt(np.abs(1 + omega_grid / coriolis_frequency)))
            )
        elif delta == 0:
            # Madsen solution
            coeff = 4 / (density * np.abs(coriolis_frequency) * mu)
            G = coeff * kv(
                0,
                2
                * sqrt(2)
                * _rot(s * np.pi / 4)
                * sqrt((z_grid / mu) * np.abs(1 + omega_grid / coriolis_frequency)),
            )
        else:
            # mixed solution
            k0z = kv(0, xiz)
            k10 = kv(1, xi0)
            coeff = (sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta
                * np.abs(coriolis_frequency)
                * density
                * sqrt(np.abs(1 + omega_grid / coriolis_frequency))
            )
            G = coeff * k0z / k10
    else:
        if mu == 0:
            # finite layer Ekman
            coeff = (sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta
                * np.abs(coriolis_frequency)
                * density
                * sqrt(np.abs(1 + omega_grid / coriolis_frequency))
            )
            argh = (
                sqrt(2)
                * _rot(s * np.pi / 4)
                * (bld / delta)
                * sqrt(np.abs(1 + omega_grid / coriolis_frequency))
            )
            argz = (
                sqrt(2)
                * _rot(s * np.pi / 4)
                * (z_grid / delta)
                * sqrt(np.abs(1 + omega_grid / coriolis_frequency))
            )

            numer = np.exp(-argz) - np.exp(argz) * np.exp(-2 * argh)
            denom = 1 + np.exp(-2 * argh)
            G = coeff * np.divide(numer, denom)

            # solution at inertial frequency
            bool_idx = omega_grid == -coriolis_frequency
            G[bool_idx] = 2 / (
                (density * np.abs(coriolis_frequency) * delta**2)
                * (bld - z_grid[bool_idx])
            )

        elif delta == 0:
            # finite layer Madsen
            coeff = 4 / (density * np.abs(coriolis_frequency) * mu)
            argz = (
                2
                * sqrt(2)
                * _rot(s * np.pi / 4)
                * sqrt((z_grid / mu) * np.abs(1 + omega_grid / coriolis_frequency))
            )
            argh = (
                2
                * sqrt(2)
                * _rot(s * np.pi / 4)
                * sqrt((bld / mu) * np.abs(1 + omega_grid / coriolis_frequency))
            )
            k0z, i0z, k0h, i0h, _, _ = _bessels_noslip(argz, argh)
            G = coeff * (k0z - i0z * k0h / i0h)

            # solution at inertial frequency
            bool_idx = omega_grid == -coriolis_frequency
            if isinstance(z_grid, np.ndarray):
                G[bool_idx] = 0.5 * coeff * np.log(bld / z_grid[bool_idx])
            else:
                G = 0.5 * coeff * np.log(bld / z_grid)
        else:
            # finite layer mixed
            G = _wind_transfer_general_no_slip(
                omega_grid, z_grid, coriolis_frequency, delta, mu, bld, density
            )

    return G


def _wind_transfer_general_no_slip(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Transfer function from wind stress to oceanic velocity with no-slip boundary condition, general form.
    """
    zo = np.divide(delta**2, mu)
    s = np.sign(coriolis_frequency) * np.sign(1 + omega / coriolis_frequency)
    xiz, xih, xi0 = _xis(s, zo, delta, z, omega, coriolis_frequency, bld)
    coeff = (
        sqrt(2)
        * _rot(-s * np.pi / 4)
        / (
            delta
            * density
            * np.abs(coriolis_frequency)
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
    )

    k0z, i0z, k0h, i0h, k10, i10 = _bessels_noslip(xiz, xih, xi0=xi0)
    numer = k0z / k10 - (k0h / k10) * (i0z / i0h)
    denom = 1 + (k0h / k10) * (i10 / i0h)
    G = coeff * np.divide(numer, denom)

    # large argument approximation
    bool_idx = np.log10(np.abs(xiz)) > 2.9
    G[bool_idx] = _wind_transfer_general_no_slip_expansion(
        omega[bool_idx], z[bool_idx], coriolis_frequency, delta, mu, bld, density
    )

    # inertial limit
    G = _wind_transfer_inertiallimit(
        G, omega, z, coriolis_frequency, delta, mu, bld, density
    )

    return G


def _wind_transfer_general_no_slip_expansion(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Compute the transfer function from wind stress to oceanic velocity with no-slip boundary, large argument approximation.
    """
    zo = np.divide(delta**2, mu)
    s = np.sign(coriolis_frequency) * np.sign(1 + omega / coriolis_frequency)
    xiz, xih, xi0 = _xis(s, zo, delta, z, omega, coriolis_frequency, bld)
    coeff = (
        sqrt(2)
        * _rot(-s * np.pi / 4)
        / (
            delta
            * density
            * np.abs(coriolis_frequency)
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
    )
    k0z, i0z, k0h, i0h, k10, i10 = _besseltildes_noslip(xiz, xih, xi0, 30)

    numer = np.exp(xi0 - xiz) * i0h * k0z - np.exp(xi0 + xiz - 2 * xih) * k0h * i0z
    denom = i0h * k10 + np.exp(2 * xi0 - 2 * xih) * k0h * i10
    G = coeff * np.divide(numer, denom)

    bool_idx = omega == -coriolis_frequency
    G[bool_idx] = (
        4
        * zo
        / (
            density
            * np.abs(coriolis_frequency)
            * delta**2
            * np.divide(sqrt(1 + bld / zo) - sqrt(1 + z / zo), (1 + z / zo) ** 0.25)
        )
    )

    return G


def _wind_transfer_inertiallimit(
    G: np.ndarray,
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_freq: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Transfer function from wind stress to oceanic velocity with no-slip boundary, inertial limit.
    """
    zo = delta**2 / mu
    bool_idx = omega == -coriolis_freq

    if np.any(bool_idx):
        if not np.isinf(bld) and not np.isinf(zo):
            G[bool_idx] = (2 / (density * np.abs(coriolis_freq) * mu)) * np.log(
                (1 + bld / zo) / (1 + z[bool_idx] / zo)
            )
        elif not np.isinf(bld) and np.isinf(zo):
            G[bool_idx] = (2 / (density * np.abs(coriolis_freq) * delta**2)) * (
                bld - z[bool_idx]
            )
        else:
            G[bool_idx] = np.inf

    return G


def _wind_transfer_free_slip(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Transfer function from wind stress to oceanic velocity with free-slip boundary condition.
    """

    zo = np.divide(delta**2, mu)

    s = np.sign(coriolis_frequency) * np.sign(1 + omega / coriolis_frequency)

    xiz, xih, xi0 = _xis(s, zo, delta, z, omega, coriolis_frequency, bld)

    if delta != 0.0 and mu != 0.0:
        coeff = (
            sqrt(2)
            * _rot(-s * np.pi / 4)
            / (
                delta
                * density
                * np.abs(coriolis_frequency)
                * sqrt(np.abs(1 + omega / coriolis_frequency))
            )
        )
        k0z, i0z, k1h, i1h, k10, i10 = _bessels_freeslip(xiz, xih, xi0=xi0)
        numer = i0z * k1h + i1h * k0z
        denom = i1h * k10 - i10 * k1h
        G = coeff * np.divide(numer, denom)

    elif mu == 0.0:
        coeff = (sqrt(2) * _rot(-s * np.pi / 4)) / (
            delta
            * np.abs(coriolis_frequency)
            * density
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
        cosharg = (
            sqrt(2)
            * _rot(s * np.pi / 4)
            * (bld - z)
            / delta
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
        sinharg = (
            sqrt(2)
            * _rot(s * np.pi / 4)
            * bld
            / delta
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
        G = coeff * np.cosh(cosharg) / np.sinh(sinharg)

    elif delta == 0.0:
        k0z, i0z, k1h, i1h, _, _ = _bessels_freeslip(xiz, xih)

        K1 = 0.5 * mu * np.abs(coriolis_frequency)

        coeff = 2 / (density * K1)

        G = coeff * (k0z + k1h * i0z / i1h)

    return G


def _wind_transfer_elipot_no_slip(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Transfer function from wind stress to oceanic velocity with no-slip boundary condition, Elipot method.
    """

    zo = np.divide(delta**2, mu)
    K0 = 0.5 * delta**2 * np.abs(coriolis_frequency)
    K1 = 0.5 * mu * np.abs(coriolis_frequency)

    delta1 = sqrt(2 * K0 / (omega + coriolis_frequency))
    delta2 = K1 / (omega + coriolis_frequency)
    xiz = 2 * sqrt(1j * (zo + z) / delta2)
    xi0 = 2 * sqrt(1j * zo / delta2)
    xih = 2 * sqrt(1j * (zo + bld) / delta2)

    if bld == np.inf:
        if K1 == 0:
            # Ekman solution
            coeff = 1 / (density * sqrt(1j * (omega + coriolis_frequency) * K0))
            G = coeff * np.exp(-z * (1 + 1j) / delta1)
        elif K0 == 0:
            # Madsen solution
            coeff = 2 / (density * K1)
            k0z = kv(0, 2 * sqrt(1j * z / delta2))
            G = coeff * k0z
        else:
            # Mixed solution
            coeff = 1 / (density * sqrt(1j * (omega + coriolis_frequency) * K0))
            G = coeff * kv(0, xiz) / kv(1, xi0)
    else:
        if K1 == 0:
            # Finite-layer Ekman
            coeff = 1 / (density * sqrt(1j * (omega + coriolis_frequency) * K0))
            numer = np.sinh((1 + 1j) * (bld - z) / delta1)
            denom = np.cosh((1 + 1j) * bld / delta1)
            G = coeff * numer / denom
        elif K0 == 0:
            # Finite-layer Madsen
            coeff = 2 / (density * K1)
            k0z = kv(0, 2 * sqrt(1j * z / delta2))
            k0h = kv(0, 2 * sqrt(1j * bld / delta2))
            i0z = iv(0, 2 * sqrt(1j * z / delta2))
            i0h = iv(0, 2 * sqrt(1j * bld / delta2))
            G = coeff * (k0z - (k0h * i0z / i0h))
        else:
            # Finite-layer mixed solution
            coeff = 1 / (density * sqrt(1j * (omega + coriolis_frequency) * K0))
            k0z, i0z, k0h, i0h, k10, i10 = _bessels_noslip(xiz, xih, xi0)
            numer = i0h * k0z - k0h * i0z
            denom = i10 * k0h + k10 * i0h
            G = coeff * numer / denom

    return G


def _wind_transfer_elipot_free_slip(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Transfer function from wind stress to oceanic velocity with free-slip boundary condition, Elipot method.
    """

    zo = np.divide(delta**2, mu)
    K0 = 0.5 * delta**2 * np.abs(coriolis_frequency)
    K1 = 0.5 * mu * np.abs(coriolis_frequency)
    delta1 = sqrt(2 * K0 / (omega + coriolis_frequency))
    delta2 = K1 / (omega + coriolis_frequency)
    xiz = 2 * sqrt(1j * (zo + z) / delta2)
    xi0 = 2 * sqrt(1j * zo / delta2)
    xih = 2 * sqrt(1j * (zo + bld) / delta2)

    if K1 == 0.0:
        # Finite-layer Ekman
        coeff = 1 / (density * sqrt(1j * (omega + coriolis_frequency) * K0))
        numer = np.cosh((1 + 1j) * (bld - z) / delta1)
        denom = np.sinh((1 + 1j) * bld / delta1)
        G = coeff * numer / denom
    elif K0 == 0.0:
        # Finite-layer Madsen
        coeff = 2 / (density * K1)
        k0z = kv(0, 2 * sqrt(1j * z / delta2))
        k0h = kv(0, 2 * sqrt(1j * bld / delta2))
        i0z = iv(0, 2 * sqrt(1j * z / delta2))
        i1h = iv(1, 2 * sqrt(1j * bld / delta2))
        G = coeff * (k0z - (k0h * i0z / i1h))
    else:
        # Finite-layer mixed solution
        coeff = 1 / (density * sqrt(1j * (omega + coriolis_frequency) * K0))
        k1h = kv(1, xih)
        i0z = iv(0, xiz)
        i1h = iv(1, xih)
        k10 = kv(1, xi0)
        i10 = iv(1, xi0)
        k0z = kv(0, xiz)
        numer = k1h * i0z + i1h * k0z
        denom = i1h * k10 - i10 * k1h
        G = coeff * numer / denom

    return G


def _bessels_freeslip(
    xiz: float | np.ndarray,
    xih: float | np.ndarray,
    xi0: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Bessel functions for the free-slip boundary condition for the xsi(z), xsi(h), and xsi(0) functions.
    """
    # Convert inputs to numpy arrays
    xiz = np.asarray(xiz)
    xih = np.asarray(xih)

    # Ensure all outputs are numpy arrays
    k0z = np.asarray(kv(0, xiz))
    i0z = np.asarray(iv(0, xiz))
    k1h = np.asarray(kv(1, xih))
    i1h = np.asarray(iv(0, xih))

    if xi0 is not None:
        xi0 = np.asarray(xi0)
        k10 = np.asarray(kv(1, xi0))
        i10 = np.asarray(iv(1, xi0))
    else:
        # Create nan values as numpy arrays with same shape as k0z
        k10 = np.full_like(k0z, np.nan)
        i10 = np.full_like(k0z, np.nan)

    return k0z, i0z, k1h, i1h, k10, i10


def _bessels_noslip(
    xiz: float | np.ndarray,
    xih: float | np.ndarray,
    xi0: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Bessel functions for the no-slip boundary condition for the xsi(z), xsi(h), and xsi(0) functions.
    """
    # Convert inputs to numpy arrays
    xiz = np.asarray(xiz)
    xih = np.asarray(xih)

    # Ensure all outputs are numpy arrays
    k0z = np.asarray(kv(0, xiz))
    i0z = np.asarray(iv(0, xiz))
    k0h = np.asarray(kv(0, xih))
    i0h = np.asarray(iv(0, xih))

    if xi0 is not None:
        xi0 = np.asarray(xi0)
        k10 = np.asarray(kv(1, xi0))
        i10 = np.asarray(iv(1, xi0))
    else:
        # Create nan values as numpy arrays with same shape as k0z
        k10 = np.full_like(k0z, np.nan)
        i10 = np.full_like(k0z, np.nan)

    return k0z, i0z, k0h, i0h, k10, i10


def _besseltildes_noslip(
    xiz: float | np.ndarray,
    xih: float | np.ndarray,
    xi0: float | np.ndarray,
    nterms: int = 30,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
]:
    """
    Compute the n-term expansion about the large-argument exponential behavior of Bessel functions
    for the xsi(z), xsi(h), and xsi(0) functions.
    """
    k0z = kvtilde(0, xiz, nterms)
    i0z = ivtilde(0, xiz, nterms)
    k0h = kvtilde(0, xih, nterms)
    i0h = ivtilde(0, xih, nterms)
    k10 = kvtilde(1, xi0, nterms)
    i10 = ivtilde(1, xi0, nterms)

    return k0z, i0z, k0h, i0h, k10, i10


def kvtilde(
    nu: int,
    z: float | np.ndarray,
    nterms: int = 30,
) -> float | np.ndarray:
    """
    Compute the n-term expansion about the large-argument exponential behavior of the modified Bessel
    function of the second kind of real order (kv).
    """
    z = np.asarray(
        z, dtype=np.complex128
    )  # Ensure z is an array for vectorized operations
    sizez = z.shape
    z = z.ravel()

    # Prepare zk for vectorized computation
    zk = np.tile(z, (nterms, 1)).T
    zk[:, 0] = 1.0
    zk = np.cumprod(zk, axis=1)

    k = np.arange(nterms)
    ak = 4.0 * nu**2 - (2.0 * k - 1.0) ** 2
    ak[0] = 1.0
    ak = np.cumprod(ak) / (factorial(k) * (8.0**k))

    # Handling potential non-finite values in ak
    if not np.all(np.isfinite(ak)):
        first_nonfinite = np.where(~np.isfinite(ak))[0][0]
        ak = ak[:first_nonfinite]
        zk = zk[:, :first_nonfinite]

    K = sqrt(np.pi / (2.0 * z)) * (np.dot(1.0 / zk, ak))
    K = K.reshape(sizez)

    return K


def ivtilde(
    nu: int,
    z: float | np.ndarray,
    nterms: int = 30,
) -> float | np.ndarray:
    """
    Compute the n-term expansion about the large-argument exponential behavior of the
    Modified Bessel function of the first kind of real order (iv).
    """
    z = np.asarray(
        z, dtype=np.complex128
    )  # Ensure z is an array for vectorized operations
    sizez = z.shape
    z = z.ravel()

    # Prepare zk for vectorized computation with alternating signs for each term
    zk = np.tile(-z, (nterms, 1)).T
    zk[:, 0] = 1.0
    zk = np.cumprod(zk, axis=1)

    k = np.arange(nterms)
    ak = 4.0 * nu**2 - (2.0 * k - 1.0) ** 2
    ak[0] = 1.0
    ak = np.cumprod(ak) / (factorial(k) * (8.0**k))

    # Handling potential non-finite values in ak
    if not np.all(np.isfinite(ak)):
        first_nonfinite = np.where(~np.isfinite(ak))[0][0]
        ak = ak[:first_nonfinite]
        zk = zk[:, :first_nonfinite]

    I = 1.0 / sqrt(2.0 * z * np.pi) * (np.dot(1.0 / zk, ak))
    I = I.reshape(sizez)

    return I


def _xis(
    s: float,
    zo: float,
    delta: float,
    z: float | np.ndarray,
    omega: float | np.ndarray,
    coriolis_frequency: float,
    bld: float,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Compute the complex-valued "xsi" functions.
    """
    xiz = (
        2
        * sqrt(2)
        * _rot(s * np.pi / 4)
        * np.divide(zo, delta)
        * sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
    )
    xih = (
        2
        * sqrt(2)
        * _rot(s * np.pi / 4)
        * np.divide(zo, delta)
        * sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
    )
    xi0 = (
        2
        * sqrt(2)
        * _rot(s * np.pi / 4)
        * np.divide(zo, delta)
        * sqrt(np.abs(1 + omega / coriolis_frequency))
    )

    return xiz, xih, xi0


def _rot(x: float) -> complex:
    """
    Compute the complex-valued rotation.
    """
    return np.exp(1j * x)

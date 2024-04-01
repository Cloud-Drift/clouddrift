"""
This module provides functions to calculate various transfer function from wind stress to oceanic
velocity.
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.lib.scimath import sqrt
from scipy.interpolate import interp1d
from scipy.special import factorial, iv, kv  # type: ignore

from clouddrift.sphere import EARTH_DAY_SECONDS


def kpp_wind_transfer(
        omega: Union[float, np.ndarray],
        cor_freq: float, 
        a: list[float], 
        zh: np.ndarray,
        zout: Optional[np.ndarray] = None,
        density: Optional[float] = 1025.0,
        ) -> np.ndarray:
    """
    Transfer function from wind stress to oceanic velocity based on the KPP model. Numerical solution.

    Parameters
    ----------
        omega: array_like or float
            Angular frequency of the wind stress forcing in radians per day.
        cor_freq: float
            Coriolis frequency, in radians per day.
        a: list[float]
            First two coefficients of the cubic profile of the eddy viscosity profile.
        zh: array_like
            Depth levels for the numerical computation of the transfer function, positive downwards, in meters.
        zout: array_like, optional
            Depths at which to return the transfer function. Default is None and the transfer function
            is returned at the same depths as the computation depth levels. Otherwise the transfer function is
            interpolated to the specified depths.
        density: float, optional, default 1025.

    Returns
    -------
        G: np.ndarray
            The transfer function from wind stress to oceanic velocity in units of kg-1 m 2 s.

    Examples
    --------
        To calculate the KPP-style transfer function:

         >>> omega = 0
         >>> z = np.linspace(0, 100, 100)
         >>> cor_freq = 2 * np.pi * 1
         >>> a = [1e-4, 1e-4]
         >>> zh = np.linspace(0, 100, 100)
         >>> G = kpp_wind_transfer(omega, cor_freq, a, zh)

    Raises
    ------
    ValueError
        If the depth levels are not positive and monotonically increasing.
    ValueError
        If the density is not positive.
    """
    # check that z is monotonic and positive
    if not np.all(np.diff(zh) > 0) and zh[0] != 0:
        raise ValueError("Depth z must be monotonically increasing from 0.")

    # check that the density is positive
    if density < 0:
        raise ValueError("Density density must be positive.")
    
    # check that all values of zout are within zh[0] and zh[-1]
    if zout is not None:
        if not (np.min(zout) >= np.min(zh) and np.max(zout) <= np.max(zh)):
            raise ValueError("All values in zout must be within the range of values in zh.")

    # convert to radians per second
    cor_freq_ = cor_freq / EARTH_DAY_SECONDS
    omega_ = np.reshape(omega/EARTH_DAY_SECONDS, (-1,))
    
    # make sure zout is an array
    if zout is not None:
        zout = np.atleast_1d(zout)

    z = np.reshape(zh, (-1,))
    h = zh[-1]

    a0 = a[0]
    a1 = a[1]
    a2 = -3 * a0 / h**2 - 2 * a1 / h
    a3 = 2 * a0 / h**3 + a1 / h**2

    up0 = -1 / (density * a0)  # du/dz at z=0
    uN = 0  # u at z=h

    N = len(z) - 1
    dz = h / N

    kappa = a0 + a1 * z + a2 * z**2 + a3 * z**3
    kappap = a1 + 2 * a2 * z + 3 * a3 * z**2
    
    if zout is None:
        G = np.zeros((N + 1, len(omega_)), dtype=complex)
    else:
        G = np.zeros((len(zout), len(omega_)), dtype=complex)

    for m, om in enumerate(omega_):
        alpha = 1j * (om + cor_freq_)
        b = np.zeros(N, dtype=complex)
        b[0] = up0 * (2 * kappa[0] / dz - kappap[0])

        D1 = kappa[1:-1] / dz**2 - kappap[1:-1] / (2 * dz)
        D2 = -alpha - 2 * kappa[:-1] / dz**2
        D3 = np.concatenate(([2 * kappa[0] / dz**2], kappa[1:-2] / dz**2 + kappap[1:-2] / (2 * dz)))

        # Construct the tridiagonal matrix A
        A = np.diag(D1, -1) + np.diag(D2, 0) + np.diag(D3, 1)

        # Solve the system A * u = b for u
        u = np.linalg.solve(A, b)
        u = np.append(u, uN)  # Append uN to the solution vector u

        if zout is None:
            G[:, m] = u
        else:
            f = interp1d(z, u, axis = 0, kind = 'linear')
            G[:, m] = f(zout)

    return G

def slab_wind_transfer(
        omega: Union[float, np.ndarray],
        cor_freq: float, 
        friction: float, 
        bld: float, 
        density: Optional[float] = 1025.0,
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
        density: float, optional
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
    omega: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    cor_freq: float,
    delta: float,
    mu: float,
    bld: float,
    boundary_condition: Optional[str] = "no-slip",
    method: Optional[str] = "lilly",
    density: Optional[float] = 1025.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        boundary_condition: str, optional
            Bottom boundary condition at the base of the ocean surface boundary layer.
            Options are "no-slip" (Default) or "free-slip".
        method: str, optional
            Method to compute the transfer function. Options are "lilly" (Default and preferred method)
            or "elipot".
        density: float, optional
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
    xiz: Union[float, np.ndarray],
    xih: Union[float, np.ndarray],
    xi0: Optional[Union[float, np.ndarray, None]] = None,
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
]:
    """
    Compute the Bessel functions for the free-slip boundary condition for the xsi(z), xsi(h), and xsi(0) functions.
    """
    k0z = kv(0, xiz)
    i0z = iv(0, xiz)
    k1h = kv(1, xih)
    i1h = iv(0, xih)

    if xi0 is not None:
        k10 = kv(1, xi0)
        i10 = iv(1, xi0)
        return k0z, i0z, k1h, i1h, k10, i10
    else:
        return k0z, i0z, k1h, i1h, np.nan, np.nan


def _bessels_noslip(
    xiz: Union[float, np.ndarray],
    xih: Union[float, np.ndarray],
    xi0: Union[float, np.ndarray, None] = None,
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
]:
    """
    Compute the Bessel functions for the no-slip boundary condition for the xsi(z), xsi(h), and xsi(0) functions.
    """
    k0z = kv(0, xiz)
    i0z = iv(0, xiz)
    k0h = kv(0, xih)
    i0h = iv(0, xih)

    if xi0 is not None:
        k10 = kv(1, xi0)
        i10 = iv(1, xi0)
        return k0z, i0z, k0h, i0h, k10, i10
    else:
        return k0z, i0z, k0h, i0h, np.nan, np.nan


def _besseltildes_noslip(
    xiz: Union[float, np.ndarray],
    xih: Union[float, np.ndarray],
    xi0: Union[float, np.ndarray],
    nterms: Optional[int] = 30,
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
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
    z: Union[float, np.ndarray],
    nterms: Optional[int] = 30,
) -> Union[float, np.ndarray]:
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
    zk[:, 0] = 1
    zk = np.cumprod(zk, axis=1)

    k = np.arange(nterms)
    ak = 4 * nu**2 - (2 * k - 1) ** 2
    ak[0] = 1
    ak = np.cumprod(ak) / (factorial(k) * (8**k))

    # Handling potential non-finite values in ak
    if not np.all(np.isfinite(ak)):
        first_nonfinite = np.where(~np.isfinite(ak))[0][0]
        ak = ak[:first_nonfinite]
        zk = zk[:, :first_nonfinite]

    K = sqrt(np.pi / (2 * z)) * (np.dot(1.0 / zk, ak))
    K = K.reshape(sizez)

    return K


def ivtilde(
    nu: int,
    z: Union[float, np.ndarray],
    nterms: Optional[int] = 30,
) -> Union[float, np.ndarray]:
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
    zk[:, 0] = 1
    zk = np.cumprod(zk, axis=1)

    k = np.arange(nterms)
    ak = 4 * nu**2 - (2 * k - 1) ** 2
    ak[0] = 1
    ak = np.cumprod(ak) / (factorial(k) * (8**k))

    # Handling potential non-finite values in ak
    if not np.all(np.isfinite(ak)):
        first_nonfinite = np.where(~np.isfinite(ak))[0][0]
        ak = ak[:first_nonfinite]
        zk = zk[:, :first_nonfinite]

    I = 1 / sqrt(2 * z * np.pi) * (np.dot(1.0 / zk, ak))
    I = I.reshape(sizez)

    return I


def _xis(
    s: float,
    zo: float,
    delta: float,
    z: Union[float, np.ndarray],
    omega: Union[float, np.ndarray],
    coriolis_frequency: float,
    bld: float,
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
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

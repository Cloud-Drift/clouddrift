"""
This module provides functions to calculate various transfer function from wind stress to oceanic
velocity.

See  Lilly, J. M. and S. Elipot (2021). A unifying perspective on transfer function solutions
to the unsteady Ekman problem. Fluids, 6 (2): 85, 1--36. doi:10.3390/fluids6020085.

See Elipot and Gille (2009), Ekman layers in the Southern Ocean: spectral models and
observations, vertical viscosity and boundary layer depth Ocean Sci., 5, 115–139, 2009
www.ocean-sci.net/5/115/2009/ doi:10.5194/os-5-115-2009
"""

from typing import Tuple, Union

import numpy as np
from scipy.special import factorial, jv, kv

# kv is the modified Bessel function of the second kind of real order v
# jv is the Bessel function of the first kind of real order v


def transfer_function(
    omega: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    cor_freq: float,
    delta: float,
    mu: float,
    bld: float,
    boundary_condition="no-slip",
    density=1025,
) -> np.ndarray:
    """
    Compute the transfer function from wind stress to oceanic velocity.

    Args:
        omega: array_like
            Angular frequency of the wind stress forcing in radians per day.
        z: array_like
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
            Bottom boundary condition. Options are "no-slip" (Default) or "free-slip".
        density: float, optional
            Water density, in kg m-3. Default is 1025.

    Returns:
        Tuple[np.ndarray, np.ndarray]
            The transfer function from wind stress to oceanic velocity.
    """
    # write a check that z is positive
    if np.any(z < 0):
        raise ValueError("Depth z must be positive.")

    # z and omega can be scalars or arrays, convert to arrays here
    z_ = np.atleast_1d(z)
    omega_ = np.atleast_1d(omega)

    # convert to radians per second
    omega_ = omega_ / 86400
    cor_freq_ = cor_freq / 86400

    # Get the lengths of 'omega' and 'z', or 1 if they are scalars
    # omega_len = len(omega_)
    # z_len = len(z_)

    # Create the transfer function array
    [omega_grid, z_grid] = np.meshgrid(omega_, z_)

    if boundary_condition == "no-slip":
        G = _transfer_function_no_slip(
            omega_grid,
            z_grid,
            cor_freq_,
            delta,
            mu,
            bld,
            density,
        )
    elif boundary_condition == "free-slip":
        G = _transfer_function_free_slip(
            omega_grid,
            z_grid,
            cor_freq_,
            delta,
            mu,
            bld,
            density,
        )

    G[z_grid > bld] = np.nan

    return G


def _transfer_function_no_slip(
    omega_grid: np.ndarray,
    z_grid: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Compute the transfer function from wind stress to oceanic velocity with no-slip boundary. Lilly version.
    """

    zo = np.divide(delta**2, mu)
    s = np.sign(coriolis_frequency) * np.sign(1 + omega_grid / coriolis_frequency)

    xiz, zih, xi0 = _xis(s, zo, delta, z_grid, omega_grid, coriolis_frequency, bld)

    if bld == np.inf:
        if mu == 0:
            # Ekman solution
            print("Ekman solution")
            coeff = (np.sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta * np.abs(coriolis_frequency) * density
            )
            G = (
                coeff
                * (
                    np.exp(
                        -(1 + s * 1j)
                        * (z_grid / delta)
                        * np.sqrt(np.abs(1 + omega_grid / coriolis_frequency))
                    )
                )
                / (np.sqrt(np.abs(1 + omega_grid / coriolis_frequency)))
            )
        elif delta == 0:
            # Madsen solution
            coeff = 4 / (density * np.abs(coriolis_frequency) * mu)
            G = coeff * kv(
                0,
                2
                * np.sqrt(2)
                * _rot(s * np.pi / 4)
                * np.sqrt((z_grid / mu) * np.abs(1 + omega_grid / coriolis_frequency)),
            )
        else:
            # mixed solution
            k0z = kv(0, xiz)
            k10 = kv(1, xi0)
            coeff = (np.sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta
                * np.abs(coriolis_frequency)
                * density
                * np.sqrt(np.abs(1 + omega_grid / coriolis_frequency))
            )
            G = coeff * k0z / k10
    else:
        if mu == 0:
            # finite layer Kkman
            coeff = (np.sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta
                * np.abs(coriolis_frequency)
                * density
                * np.sqrt(np.abs(1 + omega_grid / coriolis_frequency))
            )
            argh = (
                np.sqrt(2)
                * _rot(s * np.pi / 4)
                * (bld / delta)
                * np.sqrt(np.abs(1 + omega_grid / coriolis_frequency))
            )
            argz = (
                np.sqrt(2)
                * _rot(s * np.pi / 4)
                * (z_grid / delta)
                * np.sqrt(np.abs(1 + omega_grid / coriolis_frequency))
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
                * np.sqrt(2)
                * _rot(s * np.pi / 4)
                * np.sqrt((z_grid / mu) * np.abs(1 + omega_grid / coriolis_frequency))
            )
            argh = (
                2
                * np.sqrt(2)
                * _rot(s * np.pi / 4)
                * np.sqrt((bld / mu) * np.abs(1 + omega_grid / coriolis_frequency))
            )
            k0z, i0z, k0h, i0h, _, _ = bessels_noslip(argz, argh)
            G = coeff * (k0z - i0z * k0h / i0h)

            # solution at inertial frequency
            bool_idx = omega_grid == -coriolis_frequency
            if isinstance(z_grid, np.ndarray):
                G[bool_idx] = 0.5 * coeff * np.log(bld / z_grid[bool_idx])
            else:
                G = 0.5 * coeff * np.log(bld / z_grid)
        else:
            # finite layer mixed
            G = _transfer_function_general_no_slip(
                omega_grid, z_grid, coriolis_frequency, delta, mu, bld, density
            )

    return G  # , ddelta, dh


def _transfer_function_general_no_slip(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    """
    Compute the transfer function from wind stress to oceanic velocity with no-slip boundary, general form.
    """
    zo = np.divide(delta**2, mu)
    s = np.sign(coriolis_frequency) * np.sign(1 + omega / coriolis_frequency)
    xiz, xih, xi0 = _xis(s, zo, delta, z, omega, coriolis_frequency, bld)
    coeff = (
        np.sqrt(2)
        * _rot(-s * np.pi / 4)
        / (
            delta
            * density
            * np.abs(coriolis_frequency)
            * np.sqrt(np.abs(1 + omega / coriolis_frequency))
        )
    )

    k0z, i0z, k0h, i0h, k10, i10 = bessels_noslip(xiz, xih, xi0=xi0)
    numer = k0z / k10 - (k0h / k10) * (i0z / i0h)
    denom = 1 + (k0h / k10) * (i10 / i0h)
    G = coeff * np.divide(numer, denom)

    # large argument approximation
    bool_idx = np.log10(np.abs(xiz)) > 2.9
    G[bool_idx] = _transfer_function_general_no_slip_expansion(
        omega[bool_idx], z[bool_idx], coriolis_frequency, delta, mu, bld, density
    )

    G = _transfer_function_inertiallimit(
        G, omega, z, coriolis_frequency, delta, mu, bld, density
    )

    return G


def _transfer_function_general_no_slip_expansion(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
    zo = np.divide(delta**2, mu)
    s = np.sign(coriolis_frequency) * np.sign(1 + omega / coriolis_frequency)
    xiz, xih, xi0 = _xis(s, zo, delta, z, omega, coriolis_frequency, bld)
    coeff = (
        np.sqrt(2)
        * _rot(-s * np.pi / 4)
        / (
            delta
            * density
            * np.abs(coriolis_frequency)
            * np.sqrt(np.abs(1 + omega / coriolis_frequency))
        )
    )
    k0z, i0z, k0h, i0h, k10, i10 = besseltildes_noslip(xiz, xih, xi0, 30)

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
            * np.divide(
                np.sqrt(1 + bld / zo) - np.sqrt(1 + z / zo), (1 + z / zo) ** 0.25
            )
        )
    )

    return G


def _transfer_function_inertiallimit(
    G: np.ndarray,
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_freq: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
) -> np.ndarray:
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


def _transfer_function_free_slip(
    omega: np.ndarray,
    z: np.ndarray,
    coriolis_frequency: float,
    delta: float,
    mu: float,
    bld: float,
    density: float,
    solver: str = "general",
) -> np.ndarray:
    """
    Compute the transfer function from wind stress to oceanic velocity with free-slip boundary.
    """
    zo = np.divide(delta**2, mu)
    s = np.sign(coriolis_frequency) * np.sign(1 + omega / coriolis_frequency)

    xiz, xih, xi0 = _xis(s, zo, delta, z, omega, coriolis_frequency, bld)

    coeff = (
        np.sqrt(2)
        * _rot(-s * np.pi / 4)
        / (
            delta
            * density
            * np.abs(coriolis_frequency)
            * np.sqrt(np.abs(1 + omega / coriolis_frequency))
        )
    )
    k0z, i0z, k1h, i1h, k10, i10 = bessels_freeslip(xiz, xih, xi0=xi0)

    numer = i0z * k1h + i1h * k0z
    denom = i1h * k10 - i10 * k1h
    G = coeff * np.divide(numer, denom)

    return G


def bessels_freeslip(
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
    k0z = kv(0, xiz)
    i0z = jv(0, xiz)
    k1h = kv(1, xih)
    i1h = jv(0, xih)

    if xi0 is not None:
        k10 = kv(1, xi0)
        i10 = jv(1, xi0)
        return k0z, i0z, k1h, i1h, k10, i10
    else:
        return k0z, i0z, k1h, i1h, np.nan, np.nan


def bessels_noslip(
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
    k0z = kv(0, xiz)
    i0z = jv(0, xiz)
    k0h = kv(0, xih)
    i0h = jv(0, xih)

    if xi0 is not None:
        k10 = kv(1, xi0)
        i10 = jv(1, xi0)
        return k0z, i0z, k0h, i0h, k10, i10
    else:
        return k0z, i0z, k0h, i0h, np.nan, np.nan


def besseltildes_noslip(
    xiz: Union[float, np.ndarray],
    xih: Union[float, np.ndarray],
    xi0: Union[float, np.ndarray],
    nterms=30,
) -> Tuple[
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
]:
    k0z = besselktilde(0, xiz, nterms)
    i0z = besselitilde(0, xiz, nterms)
    k0h = besselktilde(0, xih, nterms)
    i0h = besselitilde(0, xih, nterms)
    k10 = besselktilde(1, xi0, nterms)
    i10 = besselitilde(1, xi0, nterms)

    return k0z, i0z, k0h, i0h, k10, i10


def besselktilde(
    nu: int,
    z: Union[float, np.ndarray],
    nterms=30,
) -> Union[float, np.ndarray]:
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

    K = np.sqrt(np.pi / (2 * z)) * (np.dot(1.0 / zk, ak))
    K = K.reshape(sizez)

    return K


def besselitilde(
    nu: int,
    z: Union[float, np.ndarray],
    nterms=30,
) -> Union[float, np.ndarray]:
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

    I = 1 / np.sqrt(2 * z * np.pi) * (np.dot(1.0 / zk, ak))
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
    Compute the complex-valued xi functions.
    """
    xiz = (
        2
        * np.sqrt(2)
        * _rot(s * np.pi / 4)
        * np.divide(zo, delta)
        * np.sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
    )
    xih = (
        2
        * np.sqrt(2)
        * _rot(s * np.pi / 4)
        * np.divide(zo, delta)
        * np.sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
    )
    xi0 = (
        2
        * np.sqrt(2)
        * _rot(s * np.pi / 4)
        * np.divide(zo, delta)
        * np.sqrt(np.abs(1 + omega / coriolis_frequency))
    )

    return xiz, xih, xi0


def _rot(x) -> complex:
    """
    Compute the complex-valued rotation.
    """
    return np.exp(1j * x)

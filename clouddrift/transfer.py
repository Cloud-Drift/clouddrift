"""
This module provides functions to calculate various transfer function from wind stress to oceanic
velocity.

See  Lilly, J. M. and S. Elipot (2021). A unifying perspective on transfer function solutions
to the unsteady Ekman problem. Fluids, 6 (2): 85, 1--36. doi:10.3390/fluids6020085.

See Elipot and Gille (2009), doi:.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import kv, k1, i0, i1
# kv is the modified Bessel function of the second kind of real order v


def transfer_function(
    omega: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    cor_freq: float,
    delta: float,
    mu: float,
    bld: float,
    boundary_condition="no_slip",
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

    # convert to radians per second
    omega_ = omega / 86400
    cor_freq_ = cor_freq / 86400

    # numerical parameters
    tol = 100

    # Get the lengths of 'omega' and 'z', or 1 if they are scalars
    omega_len = 1 if np.ndim(omega) == 0 else len(omega)
    z_len = 1 if np.ndim(z) == 0 else len(z)

    # Create the transfer function array
    [omega_grid, z_grid] = np.meshgrid(omega_, z)
    # G = np.zeros((omega_len, z_len), dtype=complex)

    G = _transfer_function_no_slip(
        omega_grid,
        z_grid,
        cor_freq_,
        delta,
        mu,
        bld,
        density,
    )

    return G


def _transfer_function_no_slip(
    omega: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
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
    s = np.sign(coriolis_frequency) * np.sign(1 + omega / coriolis_frequency)

    xiz, zih, xi0 = _xis(s, zo, delta, z, omega, coriolis_frequency, bld)

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
                        * (z / delta)
                        * np.sqrt(np.abs(1 + omega / coriolis_frequency))
                    )
                )
                / (np.sqrt(np.abs(1 + omega / coriolis_frequency)))
            )
        elif delta == 0:
            # Madsen solution
            coeff = 4 / (density * np.abs(coriolis_frequency) * mu)
            G = coeff * kv(
                0,
                2
                * np.sqrt(2)
                * _rot(s * np.pi / 4)
                * np.sqrt((z / mu) * np.abs(1 + omega / coriolis_frequency)),
            )
        else:
            # mixed solution
            k0z = k0(xiz)
            k10 = k1(xi0)
            coeff = (np.sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta
                * np.abs(coriolis_frequency)
                * density
                * np.sqrt(np.abs(1 + omega / coriolis_frequency))
            )
            G = coeff * k0z / k10
    else:
        if mu == 0:
            # finite layer ekman
            coeff = (np.sqrt(2) * _rot(-s * np.pi / 4)) / (
                delta
                * np.abs(coriolis_frequency)
                * density
                * np.sqrt(np.abs(1 + omega / coriolis_frequency))
            )
            argh = (
                np.sqrt(2)
                * _rot(s * np.pi / 4)
                * (bld / delta)
                * np.sqrt(np.abs(1 + omega / coriolis_frequency))
            )
            argz = (
                np.sqrt(2)
                * _rot(s * np.pi / 4)
                * (z / delta)
                * np.sqrt(np.abs(1 + omega / coriolis_frequency))
            )
            numer = np.exp(-argz) - np.exp(argz) * np.exp(-2 * argh)
            denom = 1 + np.exp(-2 * argh)
            G = coeff * numer / denom
            # stopped here. line 407 of matlab code
        elif delta == 0:
            # finite layer madsen
            G = 0
        else:
            # finite layer mixed
            G = 0

    return G  # , ddelta, dh


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

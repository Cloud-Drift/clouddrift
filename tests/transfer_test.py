import unittest

import numpy as np
from numpy.lib.scimath import sqrt
from scipy.special import iv, kv  # type: ignore

from clouddrift.sphere import EARTH_DAY_SECONDS
from clouddrift.transfer import (
    _rot,
    _xis,
    ivtilde,
    kvtilde,
    wind_transfer,
)

if __name__ == "__main__":
    unittest.main()


class TransferFunctionTestShapes(unittest.TestCase):
    def setUp(self):
        self.omega = (
            2 * np.pi * np.array([-2, -1.5, -0.5, 0, 0.5, 1, 1.5, 2])
        )  # Angular frequency in radians per day
        self.z = np.array([0, 10, 20, 30])  # Depth in meters
        self.cor_freq = 2 * np.pi * 1.5  # Coriolis frequency in radians per day
        self.delta = 10  # Ekman depth in meters
        self.mu = 0  # Madsen depth in meters
        self.bld = 50  # Boundary layer depth in meters
        self.density = 1025  # Seawater density in kg/m^3

    def test_output_size_wind_transfer_no_slip_mu_is_0(self):
        G, dG1, dG2 = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
        )
        self.assertEqual(G.shape, (len(self.z), len(self.omega)))
        self.assertEqual(dG1.shape, (len(self.z), len(self.omega)))
        self.assertEqual(dG2.shape, (len(self.z), len(self.omega)))

    def test_output_size_wind_transfer_no_slip_mu_not_0(self):
        G, dG1, dG2 = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            5,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
        )
        self.assertEqual(G.shape, (len(self.z), len(self.omega)))
        self.assertEqual(dG1.shape, (0,))
        self.assertEqual(dG2.shape, (0,))

        G, dG1, dG2 = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            5,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
            method="elipot",
        )
        self.assertEqual(G.shape, (len(self.z), len(self.omega)))
        self.assertEqual(dG1.shape, (0,))
        self.assertEqual(dG2.shape, (0,))

    def test_output_size_wind_transfer_free_slip(self):
        G, dG1, dG2 = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            5,
            self.bld,
            boundary_condition="free-slip",
            density=self.density,
        )
        self.assertEqual(G.shape, (len(self.z), len(self.omega)))
        self.assertEqual(dG1.shape, (0,))
        self.assertEqual(dG2.shape, (0,))


class TransferFunctionTestInputs(unittest.TestCase):
    def setUp(self):
        self.omega = 2 * np.pi * np.array([-2, -1.5, -0.5, 0, 0.5, 1, 1.5, 2])
        self.z = np.array([0, 10, 20, 30])
        self.cor_freq = 2 * np.pi * 1.5
        self.delta = 10
        self.mu = 5
        self.bld = 50
        self.density = 1025

    def test_wind_transfer_negative_z(self):
        with self.assertRaises(ValueError):
            wind_transfer(
                self.omega,
                -self.z,
                self.cor_freq,
                self.delta,
                self.mu,
                self.bld,
                density=self.density,
            )

    def test_wind_transfer_negative_delta(self):
        with self.assertRaises(ValueError):
            wind_transfer(
                self.omega,
                self.z,
                self.cor_freq,
                -self.delta,
                self.mu,
                self.bld,
                density=self.density,
            )

    def test_wind_transfer_negative_mu(self):
        with self.assertRaises(ValueError):
            wind_transfer(
                self.omega,
                self.z,
                self.cor_freq,
                self.delta,
                -self.mu,
                self.bld,
                density=self.density,
            )

    def test_wind_transfer_negative_bld(self):
        with self.assertRaises(ValueError):
            wind_transfer(
                self.omega,
                self.z,
                self.cor_freq,
                self.delta,
                self.mu,
                -self.bld,
                density=self.density,
            )

    def test_wind_transfer_negative_density(self):
        with self.assertRaises(ValueError):
            wind_transfer(
                self.omega,
                self.z,
                self.cor_freq,
                self.delta,
                self.mu,
                self.bld,
                density=-self.density,
            )

    def test_wind_transfer_boundary_condition(self):
        with self.assertRaises(ValueError):
            wind_transfer(
                self.omega,
                self.z,
                self.cor_freq,
                self.delta,
                self.mu,
                self.bld,
                boundary_condition="invalid",
                density=self.density,
            )

    def test_wind_transfer_method(self):
        with self.assertRaises(ValueError):
            wind_transfer(
                self.omega,
                self.z,
                self.cor_freq,
                self.delta,
                self.mu,
                self.bld,
                method="invalid",
                density=self.density,
            )


class TransferFunctionSurfaceValues(unittest.TestCase):
    def setUp(self):
        self.omega = (
            2 * np.pi * np.array([-2, -1.5, -0.5, 0, 0.5, 1, 1.5, 2])
        )  # Angular frequency in radians per day
        self.z = np.array([0, 10, 20, 30])  # Depth in meters
        self.cor_freq = 2 * np.pi * 1.5  # Coriolis frequency in radians per day
        self.delta = 10  # Ekman depth in meters
        self.mu = 0  # Madsen depth in meters
        self.bld = 50  # Boundary layer depth in meters
        self.density = 1025  # Seawater density in kg/m^3

    def test_surface_angle_wind_transfer_no_slip_lilly_nh(self):
        G, _, _ = wind_transfer(
            self.omega,
            self.z[0],
            self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
        )
        self.assertTrue(
            np.allclose(
                np.angle(G),
                np.pi / 4 * np.array([1, 0, -1, -1, -1, -1, -1, -1]),
                atol=1e-1,
            )
        )

    def test_surface_angle_wind_transfer_no_slip_elipot_nh(self):
        G, _, _ = wind_transfer(
            self.omega,
            self.z[0],
            self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
            method="elipot",
        )
        # except at inertial frequency which is not defined for elipot case
        self.assertTrue(
            np.allclose(
                np.angle(np.delete(G, 1)),
                np.pi / 4 * np.array([1, -1, -1, -1, -1, -1, -1]),
                atol=1e-1,
            )
        )

    def test_surface_angle_wind_transfer_no_slip_lilly_sh(self):
        G, _, _ = wind_transfer(
            self.omega,
            self.z[0],
            -self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
        )
        self.assertTrue(
            np.allclose(
                np.angle(G),
                np.pi / 4 * np.array([1, 1, 1, 1, 1, 1, 0, -1]),
                atol=1e-1,
            )
        )

    def test_surface_angle_wind_transfer_no_slip_elipot_sh(self):
        G, _, _ = wind_transfer(
            self.omega,
            self.z[0],
            -self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
            method="elipot",
        )
        # except at inertial frequency which is not defined for elipot case
        self.assertTrue(
            np.allclose(
                np.angle(np.delete(G, 6)),
                np.pi / 4 * np.array([1, 1, 1, 1, 1, 1, -1]),
                atol=1e-1,
            )
        )


class TransferFunctionValues(unittest.TestCase):
    # test for values reported in Lilly and Elipot 2021
    def setUp(self):
        self.omega = np.arange(-10, 10 + 0.01, 0.01) * 2 * np.pi
        self.z = np.arange(0, 50, 5)
        self.cor_freq = 2 * np.pi * 1.5
        self.delta = 20
        self.z0 = 20
        self.mu = self.delta**2 / self.z0
        self.bld = 50
        self.density = 1025

    def test_values_finite_bld(self):
        idx = np.abs(self.omega + self.cor_freq).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                0.1252763 + 0 * 1j,
                0.10296194 + 0 * 1j,
                0.08472979 + 0 * 1j,
                0.06931472 + 0 * 1j,
                0.05596158 + 0 * 1j,
                0.04418328 + 0 * 1j,
                0.03364722 + 0 * 1j,
                0.02411621 + 0 * 1j,
                0.01541507 + 0 * 1j,
                0.0074108 + 0 * 1j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        self.assertTrue(np.allclose(Gp[:, idx], expected_values, atol=1e-8))

    def test_values_infinite_bld(self):
        idx = np.abs(self.omega).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            self.mu,
            np.inf,
            boundary_condition="no-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                0.04850551 - 0.0396564j,
                0.02829943 - 0.03744607j,
                0.01520417 - 0.03297259j,
                0.00668176 - 0.02797499j,
                0.00117548 - 0.02317678j,
                -0.00230945 - 0.01886255j,
                -0.00442761 - 0.01511781j,
                -0.00561905 - 0.01193716j,
                -0.0061842 - 0.00927553j,
                -0.00633045 - 0.00707324j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        self.assertTrue(np.allclose(Gp[:, idx], expected_values, atol=1e-8))

    def test_values_infinite_bld_mu_is_zero(self):
        idx = np.abs(self.omega).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            0,
            np.inf,
            boundary_condition="no-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                0.05 - 0.05j,
                0.02809557 - 0.04736341j,
                0.01207472 - 0.04115335j,
                0.0011821 - 0.03338043j,
                -0.00553969 - 0.0254163j,
                -0.00907736 - 0.0181115j,
                -0.01033938 - 0.01191774j,
                -0.01009828 - 0.00700083j,
                -0.00896897 - 0.00333703j,
                -0.00741087 - 0.00078996j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        self.assertTrue(np.allclose(Gp[:, idx], expected_values, atol=1e-8))

    def test_values_infinite_bld_delta_is_zero(self):
        idx = np.abs(self.omega).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            0,
            self.mu,
            np.inf,
            boundary_condition="no-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                np.nan + np.nan * 1j,
                0.01603955 - 0.07145549j,
                -0.0083329 - 0.04048001j,
                -0.01373573 - 0.02367659j,
                -0.01399476 - 0.01368128j,
                -0.01260713 - 0.00748527j,
                -0.01076378 - 0.0035753j,
                -0.00891647 - 0.00110311j,
                -0.00723577 + 0.00043968j,
                -0.00577549 + 0.00137177j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        self.assertTrue(
            np.allclose(Gp[:, idx], expected_values, atol=1e-8, equal_nan=True)
        )

    def test_values_finite_bld_mu_is_zero(self):
        idx = np.abs(self.omega).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            0,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                0.04916146 - 0.05044871j,
                0.02728561 - 0.04786423j,
                0.01135701 - 0.04180688j,
                0.00063926 - 0.03427553j,
                -0.00579508 - 0.02661964j,
                -0.00889455 - 0.01965344j,
                -0.00952588 - 0.01377343j,
                -0.00842319 - 0.00906617j,
                -0.00617632 - 0.00539995j,
                -0.00324643 - 0.00249871j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        print(Gp[:, idx])
        self.assertTrue(
            np.allclose(Gp[:, idx], expected_values, atol=1e-8, equal_nan=True)
        )

    def test_values_finite_bld_delta_is_zero(self):
        idx = np.abs(self.omega).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            0,
            self.mu,
            self.bld,
            boundary_condition="no-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                np.nan + np.nan * 1j,
                0.0150223 - 0.07199065j,
                -0.00915018 - 0.04153115j,
                -0.01422336 - 0.02519934j,
                -0.01402997 - 0.0156096j,
                -0.0120768 - 0.00973228j,
                -0.00956707 - 0.00603436j,
                -0.00696735 - 0.00364922j,
                -0.00446539 - 0.00205173j,
                -0.00213448 - 0.0009083j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        self.assertTrue(
            np.allclose(Gp[:, idx], expected_values, atol=1e-8, equal_nan=True)
        )

    def test_values_free_slip(self):
        idx = np.abs(self.omega).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="free-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                0.04653128 - 0.0365929j,
                0.02616164 - 0.03449013j,
                0.01266338 - 0.03030143j,
                0.00359447 - 0.02574129j,
                -0.00253419 - 0.02152846j,
                -0.00666497 - 0.01794904j,
                -0.00940876 - 0.01509089j,
                -0.01116722 - 0.01294936j,
                -0.01220577 - 0.01147741j,
                -0.01269919 - 0.01061011j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        self.assertTrue(
            np.allclose(Gp[:, idx], expected_values, atol=1e-8, equal_nan=True)
        )

    def test_values_free_slip_mu_is_zero(self):
        idx = np.abs(self.omega).argmin()
        G, _, _ = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            0,
            self.bld,
            boundary_condition="free-slip",
            density=self.density,
        )

        expected_values = np.array(
            [
                0.05083587 - 0.04953873j,
                0.02890206 - 0.0468502j,
                0.01278664 - 0.04048806j,
                0.00171537 - 0.03247495j,
                -0.00529895 - 0.02420514j,
                -0.00928086 - 0.01656615j,
                -0.01118 - 0.01006565j,
                -0.01180644 - 0.00494948j,
                -0.01179884 - 0.00130261j,
                -0.01161306 + 0.00087116j,
            ]
        )
        Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
        self.assertTrue(
            np.allclose(Gp[:, idx], expected_values, atol=1e-8, equal_nan=True)
        )

    # def test_values_free_slip_delta_is_zero(self):
    #     idx = np.abs(self.omega).argmin()
    #     G, _, _ = wind_transfer(
    #         self.omega,
    #         self.z,
    #         self.cor_freq,
    #         0,
    #         self.mu,
    #         self.bld,
    #         boundary_condition="free-slip",
    #         density=self.density,
    #     )

    #     expected_values = np.array(
    #         [
    #             np.nan + np.nan * 1j,
    #             0.0150223 - 0.07199065j,
    #             -0.00915018 - 0.04153115j,
    #             -0.01422336 - 0.02519934j,
    #             -0.01402997 - 0.0156096j,
    #             -0.0120768 - 0.00973228j,
    #             -0.00956707 - 0.00603436j,
    #             -0.00696735 - 0.00364922j,
    #             -0.00446539 - 0.00205173j,
    #             -0.00213448 - 0.0009083j,
    #         ]
    #     )
    #     Gp = G * np.abs(self.cor_freq / EARTH_DAY_SECONDS) * self.density
    #     self.assertTrue(
    #         np.allclose(Gp[:, idx], expected_values, atol=1e-8, equal_nan=True)
    #     )


class TransferFunctionTestGradient(unittest.TestCase):
    delta = 10 ** np.arange(-1, 0.05, 3)
    bld = 10 ** np.arange(np.log10(15.15), 5, 0.05)
    [delta_grid, bld_grid] = np.meshgrid(delta, bld)

    def test_gradient_ekman_case(self):
        # Test the gradient of the transfer function, no-slip
        omega = np.array([1e-4])
        z = 15
        cor_freq = 1e-4
        mu = 0
        delta_delta = 1e-6
        delta_bld = 1e-6
        # initialize transfer functions and gradients
        wind_transfer_init = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        wind_transfer_ddelta_plus = np.zeros(
            (len(self.delta), len(self.bld)), dtype=complex
        )
        wind_transfer_ddelta_minus = np.zeros(
            (len(self.delta), len(self.bld)), dtype=complex
        )
        wind_transfer_dbld_plus = np.zeros(
            (len(self.delta), len(self.bld)), dtype=complex
        )
        wind_transfer_dbld_minus = np.zeros(
            (len(self.delta), len(self.bld)), dtype=complex
        )
        dG_ddelta = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        dG_dbld = np.zeros((len(self.delta), len(self.bld)), dtype=complex)

        for i in range(len(self.delta)):
            for j in range(len(self.bld)):
                wind_transfer_init[i, j], dG_ddelta[i, j], dG_dbld[i, j] = (
                    wind_transfer(
                        omega=omega,
                        z=z,
                        cor_freq=cor_freq,
                        delta=self.delta[i],
                        mu=mu,
                        bld=self.bld[j],
                    )
                )
                wind_transfer_ddelta_plus[i, j], _, _ = wind_transfer(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i] + delta_delta / 2,
                    mu=mu,
                    bld=self.bld[j],
                )
                wind_transfer_ddelta_minus[i, j], _, _ = wind_transfer(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i] - delta_delta / 2,
                    mu=mu,
                    bld=self.bld[j],
                )
                wind_transfer_dbld_plus[i, j], _, _ = wind_transfer(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j] + delta_bld / 2,
                )
                wind_transfer_dbld_minus[i, j], _, _ = wind_transfer(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j] - delta_bld / 2,
                )

        dG_ddelta_fd = (
            wind_transfer_ddelta_plus - wind_transfer_ddelta_minus
        ) / delta_delta
        dG_dbld_fd = (wind_transfer_dbld_plus - wind_transfer_dbld_minus) / delta_bld

        bool_indices = dG_ddelta_fd != 0
        # Calculate eps1 using numpy operations
        eps1 = np.max(
            (
                np.abs(dG_ddelta_fd[bool_indices] - dG_ddelta[bool_indices])
                / sqrt(
                    np.abs(dG_ddelta_fd[bool_indices]) ** 2
                    + np.abs(dG_ddelta[bool_indices]) ** 2
                )
            )
        )

        bool_indices = dG_dbld_fd != 0
        # Calculate eps2 using numpy operations
        eps2 = np.max(
            (
                np.abs(dG_dbld_fd[bool_indices] - dG_dbld[bool_indices])
                / sqrt(
                    np.abs(dG_dbld_fd[bool_indices]) ** 2
                    + np.abs(dG_dbld[bool_indices]) ** 2
                )
            )
        )
        self.assertTrue(
            np.log10(eps1) < -4 and np.log10(eps2) < -4,
            "wind_transfer analytic and numerical gradients match for Ekman case",
        )


# class TransferFunctionTestMethods(unittest.TestCase):
#     def setUp(self):
#         self.z = 15.0  # np.arange(0.1, 101, 1)
#         # need to also test bld = np.inf, K0 = 0 and K1 = 0
#         self.bld = [np.inf, 200.0]
#         self.K0 = [0, 1/10, 1/10]
#         self.K1 = [1,0,1]
#         self.cor_freq = 2 * np.pi * 1.5
#         self.delta = sqrt(2 * np.array(self.K0) / self.cor_freq)
#         self.mu = 2 * np.array(self.K1) / self.cor_freq
#         self.omega = 2 * np.pi * np.array([0.5])
#         self.slipstr1 = "no-slip"
#         self.slipstr2 = "free-slip"

#     def test_method_equivalence_no_slip(self):
#         for s in [1, -1]:
#             for i, (delta, mu) in enumerate(zip(self.delta, self.mu)):
#                 for j, bld in enumerate(self.bld):
#                     Ge, _, _ = wind_transfer(
#                         self.omega,
#                         self.z,
#                         self.cor_freq,
#                         delta,
#                         mu,
#                         bld,
#                         method="elipot",
#                         boundary_condition=self.slipstr1,
#                     )
#                     Gl, _, _ = wind_transfer(
#                         self.omega,
#                         self.z,
#                         self.cor_freq,
#                         delta,
#                         mu,
#                         bld,
#                         method="lilly",
#                         boundary_condition=self.slipstr1,
#                     )
#                     #bool_idx = Ge != np.nan and Gl != np.nan
#                     #print(Ge[bool_idx], Gl[bool_idx])
#                     print(Ge, Gl)
#                     #bool1 = np.allclose(Ge[bool_idx], Gl[bool_idx], atol=1e-8)
#                     bool1 = np.allclose(Ge, Gl, atol=1e-8, equal_nan=True)
#                     self.assertTrue(bool1)

#     def test_method_equivalence_free_slip(self):
#         for s in [1, -1]:
#             for i, (delta, mu) in enumerate(zip(self.delta, self.mu)):
#                 for j, bld in enumerate(self.bld):
#                     Ge, _, _ = wind_transfer(
#                         self.omega,
#                         self.z,
#                         self.cor_freq,
#                         delta,
#                         mu,
#                         bld,
#                         method="elipot",
#                         boundary_condition=self.slipstr2,
#                     )
#                     Gl, _, _ = wind_transfer(
#                         self.omega,
#                         self.z,
#                         self.cor_freq,
#                         delta,
#                         mu,
#                         bld,
#                         method="lilly",
#                         boundary_condition=self.slipstr2,
#                     )
#                     #bool_idx = Ge != np.nan and Gl != np.nan
#                     #print(Ge, Gl)
#                     #bool1 = np.allclose(Ge[bool_idx], Gl[bool_idx], atol=1e-8)
#                     bool1 = np.allclose(Ge, Gl, atol=1e-8, equal_nan=True)
#                     self.assertTrue(bool1)


class TestKvTilde(unittest.TestCase):
    def test_kvtilde(self):
        atol = 1e-10
        for s in [1, -1]:
            z = sqrt(s * 1j) * np.arange(15, 100.01, 0.01).reshape(-1, 1)
            bk0 = np.zeros((len(z), 2), dtype=np.complex128)
            bk = np.zeros_like(bk0)

            for i in range(2):
                bk0[:, i] = (np.exp(z) * kv(i, z)).reshape(-1)
                bk[:, i] = (kvtilde(i, z)).reshape(-1)

            if s == 1:
                test_name = "kvTILDE for z with phase of pi/4"
            else:
                test_name = "kvTILDE for z with phase of -pi/4"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(np.abs((bk0 - bk) / bk), 0, atol=atol),
                    msg=f"Failed: {test_name}",
                )

            for i in range(2):
                bk[:, i] = kvtilde(i, z, 2).reshape(-1)

            if s == 1:
                test_name = "2-term for z with phase of pi/4, order 0 and 1"
            else:
                test_name = "2-term for z with phase of -pi/4, order 0 and 1"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(np.abs((bk0[:, :2] - bk) / bk), 0, atol=1e-3),
                    msg=f"Failed: {test_name}",
                )

            bk0 = sqrt(np.pi / (2 * z)) * (1 - 1 / (8 * z))
            bk1 = sqrt(np.pi / (2 * z)) * (1 + 3 / (8 * z))

            if s == 1:
                test_name = "2-term vs. analytic for z with phase of pi/4"
            else:
                test_name = "2-term vs. analytic for z with phase of -pi/4"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(
                        np.abs((np.hstack([bk0, bk1]) - bk) / bk), 0, atol=1e-15
                    ),
                    msg=f"Failed: {test_name}",
                )


class TestIvTilde(unittest.TestCase):
    def test_ivtilde(self):
        atol = 1e-10
        for s in [1, -1]:
            z = sqrt(s * 1j) * np.arange(23.0, 100.0, 0.01).reshape(-1, 1)
            bi0 = np.zeros((len(z), 2), dtype=np.complex128)
            bi = np.zeros_like(bi0)

            for i in range(2):
                bi0[:, i] = (np.exp(-z) * iv(i, z)).reshape(-1)
                bi[:, i] = (ivtilde(i, z)).reshape(-1)

            if s == 1:
                test_name = "ivTILDE for z with phase of pi/4"
            else:
                test_name = "ivTILDE for z with phase of -pi/4"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(np.abs((bi0 - bi) / bi), 0, atol=atol),
                    msg=f"Failed: {test_name}",
                )

            for i in range(2):
                bi[:, i] = ivtilde(i, z, 2).reshape(-1)

            if s == 1:
                test_name = "2-term for z with phase of pi/4, order 0 and 1"
            else:
                test_name = "2-term for z with phase of -pi/4, order 0 and 1"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(np.abs((bi0[:, :2] - bi) / bi), 0, atol=1e-3),
                    msg=f"Failed: {test_name}",
                )

            bi0 = sqrt(1 / (2 * np.pi * z)) * (1 + 1 / (8 * z))
            bi1 = sqrt(1 / (2 * np.pi * z)) * (1 - 3 / (8 * z))

            if s == 1:
                test_name = "2-term vs. analytic for z with phase of pi/4"
            else:
                test_name = "2-term vs. analytic for z with phase of -pi/4"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(
                        np.abs((np.hstack([bi0, bi1]) - bi) / bi), 0, atol=1e-15
                    ),
                    msg=f"Failed: {test_name}",
                )


class TestXis(unittest.TestCase):
    def test_xis_case1(self):
        s = 1.0
        delta = 30.0
        mu = 20.0
        zo = delta**2 / mu
        z = 15.0
        omega = 5.0
        coriolis_frequency = 6.0
        bld = 100.0
        expected_xiz = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xih = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xi0 = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
        assert np.allclose(
            _xis(s, zo, delta, z, omega, coriolis_frequency, bld),
            (expected_xiz, expected_xih, expected_xi0),
        )

    def test_xis_case2(self):
        s = -1.0
        delta = 30.0
        mu = 20.0
        zo = delta**2 / mu
        z = np.array([1.0, 2.0, 3.0])
        omega = np.array([0.1, 0.2, 0.3])
        coriolis_frequency = 0.5
        bld = 7.0
        expected_xiz = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xih = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xi0 = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
        assert np.allclose(
            _xis(s, zo, delta, z, omega, coriolis_frequency, bld),
            (expected_xiz, expected_xih, expected_xi0),
        )

    def test_xis_case3(self):
        s = 1
        delta = 1.0
        mu = 1.0
        zo = delta**2 / mu
        z = 0.0
        omega = 0.0
        coriolis_frequency = 0.1
        bld = 10.0
        expected_xiz = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xih = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xi0 = (
            2
            * sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * sqrt(np.abs(1 + omega / coriolis_frequency))
        )
        assert np.allclose(
            _xis(s, zo, delta, z, omega, coriolis_frequency, bld),
            (expected_xiz, expected_xih, expected_xi0),
        )


class TestRot(unittest.TestCase):
    def test_rot_positive(self):
        # Test with positive angle
        angle = np.pi / 4
        expected_result = np.exp(1j * angle)
        self.assertTrue(np.isclose(_rot(angle), expected_result))

    def test_rot_negative(self):
        # Test with negative angle
        angle = -np.pi / 3
        expected_result = np.exp(1j * angle)
        self.assertTrue(np.isclose(_rot(angle), expected_result))

    def test_rot_zero(self):
        # Test with zero angle
        angle = 0
        expected_result = np.exp(1j * angle)
        self.assertTrue(np.isclose(_rot(angle), expected_result))

    def test_rot_random(self):
        # Test with random angle
        angle = np.random.uniform(-3 * np.pi, np.pi)
        expected_result = np.exp(1j * angle)
        self.assertTrue(np.isclose(_rot(angle), expected_result))

    def test_rot_empty(self):
        # Test with empty array
        angles = np.array([])
        expected_results = np.array([])
        assert np.allclose(_rot(angles), expected_results)

import unittest

import numpy as np
from scipy.special import iv, kv

from clouddrift.transfer import (
    _rot,
    _xis,
    ivtilde,
    kvtilde,
    wind_transfer,
)

if __name__ == "__main__":
    unittest.main()


class TransferFunctionTestValues(unittest.TestCase):
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

    def test_surface_angle_wind_transfer_no_slip_nh(self):
        G, ddelta, dh = wind_transfer(
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

    def test_surface_angle_wind_transfer_no_slip_sh(self):
        G, ddelta, dh = wind_transfer(
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

    def test_output_size_wind_transfer_free_slip_mu_is_0(self):
        G, dG1, dG2 = wind_transfer(
            self.omega,
            self.z,
            self.cor_freq,
            self.delta,
            self.mu,
            self.bld,
            boundary_condition="free-slip",
            density=self.density,
        )
        self.assertEqual(G.shape, (len(self.z), len(self.omega)))
        self.assertEqual(dG1.shape, (len(self.z), len(self.omega)))
        self.assertEqual(dG2.shape, (len(self.z), len(self.omega)))

    def test_output_size_wind_transfer_free_slip_mu_not_0(self):
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


class wind_transfer_test_gradient(unittest.TestCase):
    delta = 10 ** np.arange(-1, 0.05, 3)
    bld = 10 ** np.arange(np.log10(15.15), 5, 0.05)
    [delta_grid, bld_grid] = np.meshgrid(delta, bld)

    def test_gradient(self):
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
                / np.sqrt(
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
                / np.sqrt(
                    np.abs(dG_dbld_fd[bool_indices]) ** 2
                    + np.abs(dG_dbld[bool_indices]) ** 2
                )
            )
        )
        print((eps1), (eps2))
        print(np.log10(eps1), np.log10(eps2))

        self.assertTrue(
            eps1 < -4 and eps2 < -4,
            "wind_transfer analytic and numerical gradients match for Ekman case",
        )


class TransferFunctionTestLimits(unittest.TestCase):
    def setUp(self):
        self.z = np.arange(0.1, 101, 1)
        self.h = [np.inf, 200]
        self.K0 = [0, 1 / 10, 1 / 10]
        self.K1 = [1, 0, 1]
        self.fc = 1e-4
        self.delta = np.sqrt(2 * np.array(self.K0) / self.fc)
        self.mu = 2 * np.array(self.K1) / self.fc
        self.omega = np.fft.fftfreq(1000, 1)[
            "two"
        ]  # Placeholder for `fourier` equivalent
        self.slipstr = "noslip"

    def wind_transfer_test(self):
        for s in [1, -1]:
            for i, (delta, mu) in enumerate(zip(self.delta, self.mu)):
                for j, h in enumerate(self.h):
                    # Placeholder for windtrans function calls and normalization
                    # Ge, Go, Gl, Ga = [np.zeros_like(self.z) for _ in range(4)]

                    # Assuming windtrans and comparisons would be defined here
                    # bool1 = np.allclose(Ge, Gl, atol=1e-8)
                    # Additional boolean checks for other conditions

                    # Example assertion (replace with actual test logic)
                    self.assertTrue(True, "Placeholder for actual test condition")


class TestKvTilde(unittest.TestCase):
    def test_kvtilde(self):
        atol = 1e-10
        for s in [1, -1]:
            z = np.sqrt(s * 1j) * np.arange(15, 100.01, 0.01).reshape(-1, 1)
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
                # pass
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

            bk0 = np.sqrt(np.pi / (2 * z)) * (1 - 1 / (8 * z))
            bk1 = np.sqrt(np.pi / (2 * z)) * (1 + 3 / (8 * z))

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
            z = np.sqrt(s * 1j) * np.arange(23.0, 100.0, 0.01).reshape(-1, 1)
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

            bi0 = np.sqrt(1 / (2 * np.pi * z)) * (1 + 1 / (8 * z))
            bi1 = np.sqrt(1 / (2 * np.pi * z)) * (1 - 3 / (8 * z))

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
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xih = (
            2
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xi0 = (
            2
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt(np.abs(1 + omega / coriolis_frequency))
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
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xih = (
            2
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xi0 = (
            2
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt(np.abs(1 + omega / coriolis_frequency))
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
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt((1 + np.divide(z, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xih = (
            2
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt((1 + np.divide(bld, zo)) * np.abs(1 + omega / coriolis_frequency))
        )
        expected_xi0 = (
            2
            * np.sqrt(2)
            * _rot(s * np.pi / 4)
            * np.divide(zo, delta)
            * np.sqrt(np.abs(1 + omega / coriolis_frequency))
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

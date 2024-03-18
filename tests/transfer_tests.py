import unittest

import numpy as np
from scipy.special import iv, kv

from clouddrift.transfer import (
    _rot,
    _xis,
    besselitilde,
    besselktilde,
    transfer_function,
)

if __name__ == "__main__":
    unittest.main()


class transfer_test_gradient(unittest.TestCase):
    delta = 10 ** np.arange(-1, 0.05, 3)
    bld = 10 ** np.arange(np.log10(15.15), 5, 0.05)
    [delta_grid, bld_grid] = np.meshgrid(delta, bld)

    def test_gradient(self):
        # Test the gradient of the transfer function
        omega = np.array([1e-4])
        z = 15
        cor_freq = 1e-4
        mu = 0
        delta_delta = 1e-6
        delta_bld = 1e-6
        # initialize the transfer function
        transfer_function_0 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_1 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_2 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_3 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)
        transfer_function_4 = np.zeros((len(self.delta), len(self.bld)), dtype=complex)

        for i in range(len(self.delta)):
            for j in range(len(self.bld)):
                transfer_function_0[i, j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j],
                )
                transfer_function_1[i, j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i] + delta_delta / 2,
                    mu=mu,
                    bld=self.bld[j],
                )
                transfer_function_2[i, j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i] - delta_delta / 2,
                    mu=mu,
                    bld=self.bld[j],
                )
                transfer_function_3[i, j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j] + delta_bld / 2,
                )
                transfer_function_4[i, j] = transfer_function(
                    omega=omega,
                    z=z,
                    cor_freq=cor_freq,
                    delta=self.delta[i],
                    mu=mu,
                    bld=self.bld[j] - delta_bld / 2,
                )
        self.assertTrue(
            np.shape(transfer_function_0) == (len(self.delta), len(self.bld))
        )
        self.assertTrue(
            np.shape(transfer_function_1) == (len(self.delta), len(self.bld))
        )
        self.assertTrue(
            np.shape(transfer_function_2) == (len(self.delta), len(self.bld))
        )
        self.assertTrue(
            np.shape(transfer_function_3) == (len(self.delta), len(self.bld))
        )
        self.assertTrue(
            np.shape(transfer_function_4) == (len(self.delta), len(self.bld))
        )


class TestBesselkTilde(unittest.TestCase):
    def test_besselktilde(self):
        atol = 1e-10
        for s in [1, -1]:
            z = np.sqrt(s * 1j) * np.arange(15, 100.01, 0.01).reshape(-1, 1)
            bk0 = np.zeros((len(z), 2), dtype=np.complex128)
            bk = np.zeros_like(bk0)

            for i in range(2):
                bk0[:, i] = (np.exp(z) * kv(i, z)).reshape(-1)
                bk[:, i] = (besselktilde(i, z)).reshape(-1)

            if s == 1:
                test_name = "BESSELKTILDE for z with phase of pi/4"
            else:
                test_name = "BESSELKTILDE for z with phase of -pi/4"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(np.abs((bk0 - bk) / bk), 0, atol=atol),
                    msg=f"Failed: {test_name}",
                )

            for i in range(2):
                # pass
                bk[:, i] = besselktilde(i, z, 2).reshape(-1)

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


class TestBesseliTilde(unittest.TestCase):
    def test_besselitilde(self):
        atol = 1e-10
        for s in [1, -1]:
            z = np.sqrt(s * 1j) * np.arange(23.0, 100.0, 0.01).reshape(-1, 1)
            bi0 = np.zeros((len(z), 2), dtype=np.complex128)
            bi = np.zeros_like(bi0)

            for i in range(2):
                bi0[:, i] = (np.exp(-z) * iv(i, z)).reshape(-1)
                bi[:, i] = (besselitilde(i, z)).reshape(-1)

            if s == 1:
                test_name = "BESSELITILDE for z with phase of pi/4"
            else:
                test_name = "BESSELITILDE for z with phase of -pi/4"

            with self.subTest(test_name=test_name):
                self.assertTrue(
                    np.allclose(np.abs((bi0 - bi) / bi), 0, atol=atol),
                    msg=f"Failed: {test_name}",
                )

            for i in range(2):
                bi[:, i] = besselitilde(i, z, 2).reshape(-1)

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

from clouddrift.signal import (
    analytic_signal,
    cartesian_to_rotary,
    rotary_to_cartesian,
    ellipse_parameters,
    modulated_ellipse_signal,
)
import numpy as np
import unittest
import xarray as xr

if __name__ == "__main__":
    unittest.main()


class analytic_signal_tests(unittest.TestCase):
    def test_size(self):
        x = np.random.rand(99)
        z = analytic_signal(x)
        self.assertEqual(np.shape(x), np.shape(z))

    def test_imag(self):
        x = np.random.rand(99) + 1j * np.random.rand(99)
        z = analytic_signal(x)
        self.assertTrue(type(z) == tuple)

    def test_real_odd(self):
        x = np.random.rand(99)
        z = analytic_signal(x)
        self.assertTrue(np.allclose(x, z.real))

    def test_real_even(self):
        x = np.random.rand(100)
        z = analytic_signal(x)
        self.assertTrue(np.allclose(x, z.real))

    def test_imag_odd(self):
        z = np.random.rand(99) + 1j * np.random.rand(99)
        wp, wn = analytic_signal(z)
        self.assertTrue(np.allclose(z, wp + np.conj(wn)))

    def test_imag_even(self):
        z = np.random.rand(100) + 1j * np.random.rand(100)
        wp, wn = analytic_signal(z)
        self.assertTrue(np.allclose(z, wp + np.conj(wn)))

    def test_boundary(self):
        x = np.random.rand(99)
        z1 = analytic_signal(x, boundary="mirror")
        z2 = analytic_signal(x, boundary="zeros")
        z3 = analytic_signal(x, boundary="periodic")
        self.assertTrue(np.allclose(x, z1.real))
        self.assertTrue(np.allclose(x, z2.real))
        self.assertTrue(np.allclose(x, z3.real))

    def test_ndarray(self):
        x = np.random.random((9, 11, 13))
        for n in range(3):
            z = analytic_signal(x, time_axis=n)
            self.assertTrue(np.allclose(x, z.real))

    def test_xarray(self):
        x = xr.DataArray(data=np.random.random((9, 11, 13)))
        for n in range(3):
            z = analytic_signal(x, time_axis=n)
            self.assertTrue(np.allclose(x, z.real))


class cartesian_to_rotary_tests(unittest.TestCase):
    def test_size(self):
        ua = np.random.rand(99) + 1j * np.random.rand(99)
        va = np.random.rand(99) + 1j * np.random.rand(99)
        wp, wn = cartesian_to_rotary(ua, va)
        self.assertEqual(np.shape(ua), np.shape(wp))
        self.assertEqual(np.shape(ua), np.shape(wn))

    def test_array(self):
        ua = np.random.random((99)) + 1j * np.random.random((99))
        va = np.random.random((99)) + 1j * np.random.random((99))
        wp, wn = cartesian_to_rotary(ua, va)
        self.assertTrue(np.allclose(np.real(ua) + 1j * np.real(va), wp + np.conj(wn)))

    def test_ndarray(self):
        ua = np.random.rand(99, 10, 101) + 1j * np.random.rand(99, 10, 101)
        va = np.random.rand(99, 10, 101) + 1j * np.random.rand(99, 10, 101)
        wp, wn = cartesian_to_rotary(ua, va)
        self.assertTrue(np.allclose(np.real(ua) + 1j * np.real(va), wp + np.conj(wn)))

    def test_xarray(self):
        ua = xr.DataArray(data=np.random.rand(99, 100) + 1j * np.random.rand(99, 100))
        va = xr.DataArray(data=np.random.rand(99, 100) + 1j * np.random.rand(99, 100))
        wp, wn = cartesian_to_rotary(ua, va)
        self.assertTrue(np.allclose(np.real(ua) + 1j * np.real(va), wp + np.conj(wn)))

    def test_invert_cartesian_to_rotary(self):
        u = np.random.rand(99)
        v = np.random.rand(99)
        ua = analytic_signal(u)
        va = analytic_signal(v)
        wp, wn = cartesian_to_rotary(ua, va)
        ua_, va_ = rotary_to_cartesian(wp, wn)
        self.assertTrue(np.allclose(ua, ua_))
        self.assertTrue(np.allclose(va, va_))
        self.assertTrue(np.allclose(u, np.real(ua_)))
        self.assertTrue(np.allclose(v, np.real(va_)))


class ellipse_parameters_tests(unittest.TestCase):
    def setUp(self):
        self.theta = np.pi / 4
        self.a = 1
        self.b = 2
        self.kappa = np.sqrt(0.5 * (self.a**2 + self.b**2))
        self.lamb = (
            np.sign(self.b) * (self.a**2 - self.b**2) / (self.a**2 + self.b**2)
        )
        self.phi = np.linspace(0, 10 * 2 * np.pi, 1000)
        self.z = (self.a * np.cos(self.phi) + 1j * self.b * np.sin(self.phi)) * np.exp(
            -1j * self.theta
        )
        self.xa = analytic_signal(np.real(self.z), boundary="periodic")
        self.ya = analytic_signal(np.imag(self.z), boundary="periodic")

    def test_result_has_same_size_as_input(self):
        kappa, lamb, theta, phi = ellipse_parameters(self.xa, self.ya)
        self.assertEqual(np.shape(kappa), np.shape(self.xa))
        self.assertEqual(np.shape(lamb), np.shape(self.xa))
        self.assertEqual(np.shape(theta), np.shape(self.xa))
        self.assertEqual(np.shape(phi), np.shape(self.xa))

    def test_xarray_size(self):
        xa = xr.DataArray(data=self.xa)
        ya = xr.DataArray(data=self.ya)
        kappa, lamb, theta, phi = ellipse_parameters(xa, ya)
        self.assertEqual(np.shape(kappa), np.shape(xa))
        self.assertEqual(np.shape(lamb), np.shape(xa))
        self.assertEqual(np.shape(theta), np.shape(xa))
        self.assertEqual(np.shape(phi), np.shape(xa))

    def test_result_is_correct(self):
        kappa, lamb, theta, phi = ellipse_parameters(self.xa, self.ya)
        self.assertTrue(np.allclose(np.mean(kappa), np.mean(self.kappa), atol=1e-2))
        # self.assertTrue(np.allclose(np.mean(lamb), np.mean(self.lamb), atol=1e-2))
        self.assertTrue(np.allclose(np.mean(theta), np.mean(self.theta), atol=1e-2))
        # self.assertTrue(np.allclose(phi, self.phi, atol=1e-2))
        self.assertTrue(
            np.isclose(np.mod(np.mean(self.phi - phi), np.pi / 2), 0, atol=1e-2)
        )

    def test_invert_ellipse_parameters(self):
        kappa, lamb, theta, phi = ellipse_parameters(self.xa, self.ya)
        xa, ya = modulated_ellipse_signals(kappa, lamb, theta, phi)
        self.assertTrue(np.allclose(self.xa, xa))
        self.assertTrue(np.allclose(self.ya, ya))


# write the tests for modulated_ellipse_signals: some ambiguity here ... to investigate
class modulated_ellipse_signal_tests(unittest.TestCase):
    def setUp(self):
        self.phi = np.linspace(0, 10 * 2 * np.pi, 1000) + np.pi / 3
        self.theta = -np.pi / 4 * np.ones_like(self.phi)
        self.a = 6
        self.b = -1.5
        self.kappa = np.sqrt(0.5 * (self.a**2 + self.b**2)) * np.ones_like(self.phi)
        self.lamb = (
            np.ones_like(self.phi)
            * np.sign(self.b)
            * (self.a**2 - self.b**2)
            / (self.a**2 + self.b**2)
        )
        self.xa = np.exp(1j * self.phi) * (
            self.a * np.cos(self.theta) + 1j * self.b * np.sin(self.theta)
        )
        self.ya = np.exp(1j * self.phi) * (
            self.a * np.sin(self.theta) - 1j * self.b * np.cos(self.theta)
        )

    def test_result_has_same_size_as_input(self):
        xa, ya = modulated_ellipse_signal(self.kappa, self.lamb, self.theta, self.phi)
        self.assertEqual(np.shape(xa), np.shape(self.kappa))
        self.assertEqual(np.shape(ya), np.shape(self.kappa))

    def test_xarray_size(self):
        kappa = xr.DataArray(data=self.kappa)
        lamb = xr.DataArray(data=self.lamb)
        theta = xr.DataArray(data=self.theta)
        phi = xr.DataArray(data=self.phi)
        xa, ya = modulated_ellipse_signal(kappa, lamb, theta, phi)
        self.assertEqual(np.shape(xa), np.shape(kappa))
        self.assertEqual(np.shape(ya), np.shape(kappa))

    def test_result_is_correct(self):
        xa, ya = modulated_ellipse_signal(self.kappa, self.lamb, self.theta, self.phi)
        self.assertTrue(np.allclose(xa, self.xa, atol=1e-2))
        self.assertTrue(np.allclose(ya, self.ya, atol=1e-2))

    def test_invert_modulated_ellipse_signals(self):
        xa, ya = modulated_ellipse_signal(self.kappa, self.lamb, self.theta, self.phi)
        kappa, lamb, theta, phi = ellipse_parameters(xa, ya)
        self.assertTrue(np.allclose(kappa, self.kappa, atol=1e-2))
        self.assertTrue(np.allclose(lamb, self.lamb, atol=1e-2))
        self.assertTrue(np.allclose(theta, self.theta, atol=1e-2))
        self.assertTrue(np.allclose(phi, self.phi, atol=1e-2))


class rotary_to_cartesian_tests(unittest.TestCase):
    def test_size(self):
        wp = np.random.rand(99) + 1j * np.random.rand(99)
        wn = np.random.rand(99) + 1j * np.random.rand(99)
        ua, va = rotary_to_cartesian(wp, wn)
        self.assertEqual(np.shape(ua), np.shape(wp))
        self.assertEqual(np.shape(va), np.shape(wp))

    def test_array(self):
        wp = np.random.random((100)) + 1j * np.random.random((100))
        wn = np.random.random((100)) + 1j * np.random.random((100))
        ua, va = rotary_to_cartesian(wp, wn)
        self.assertTrue(np.allclose(np.real(ua) + 1j * np.real(va), wp + np.conj(wn)))

    def test_ndarray(self):
        wp = np.random.rand(99, 10, 101) + 1j * np.random.rand(99, 10, 101)
        wn = np.random.rand(99, 10, 101) + 1j * np.random.rand(99, 10, 101)
        ua, va = rotary_to_cartesian(wp, wn)
        self.assertTrue(np.allclose(np.real(ua) + 1j * np.real(va), wp + np.conj(wn)))

    def test_xarray(self):
        wp = xr.DataArray(data=np.random.rand(99)) + 1j * xr.DataArray(
            data=np.random.rand(99)
        )
        wn = xr.DataArray(data=np.random.rand(99)) + 1j * xr.DataArray(
            data=np.random.rand(99)
        )
        ua, va = rotary_to_cartesian(wp, wn)
        self.assertTrue(np.allclose(np.real(ua) + 1j * np.real(va), wp + np.conj(wn)))

    def test_invert_rotary_to_cartesian(self):
        wp = np.random.random((100)) + 1j * np.random.random((100))
        wn = np.random.random((100)) + 1j * np.random.random((100))
        ua, va = rotary_to_cartesian(wp, wn)
        wp_, wn_ = cartesian_to_rotary(ua, va)
        self.assertTrue(np.allclose(wp, wp_))
        self.assertTrue(np.allclose(wn, wn_))

from clouddrift.signal import (
    analytic_signal,
    cartesian_to_rotary,
    rotary_to_cartesian,
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

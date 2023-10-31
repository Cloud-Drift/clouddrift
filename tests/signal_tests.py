from clouddrift.signal import (
    analytic_transform,
    cartesian_to_rotary,
    rotary_to_cartesian,
)
import numpy as np
import unittest
import xarray as xr

if __name__ == "__main__":
    unittest.main()


class analytic_transform_tests(unittest.TestCase):
    def test_real_odd(self):
        x = np.random.rand(99)
        z = analytic_transform(x)
        self.assertTrue(np.allclose(x - np.mean(x), z.real))

    def test_real_even(self):
        x = np.random.rand(100)
        z = analytic_transform(x)
        self.assertTrue(np.allclose(x - np.mean(x), z.real))

    def test_imag_odd(self):
        z = np.random.rand(99) + 1j * np.random.rand(99)
        zp = analytic_transform(z)
        zn = analytic_transform(np.conj(z))
        self.assertTrue(np.allclose(z - np.mean(z, keepdims=True), zp + np.conj(zn)))

    def test_imag_even(self):
        z = np.random.rand(100) + 1j * np.random.rand(100)
        zp = analytic_transform(z)
        zn = analytic_transform(np.conj(z))
        self.assertTrue(np.allclose(z - np.mean(z, keepdims=True), zp + np.conj(zn)))

    def test_boundary(self):
        x = np.random.rand(99)
        z1 = analytic_transform(x, boundary="mirror")
        z2 = analytic_transform(x, boundary="zeros")
        z3 = analytic_transform(x, boundary="periodic")
        self.assertTrue(np.allclose(x - np.mean(x), z1.real))
        self.assertTrue(np.allclose(x - np.mean(x), z2.real))
        self.assertTrue(np.allclose(x - np.mean(x), z3.real))

    def test_ndarray(self):
        x = np.random.random((9, 11, 13))
        for n in range(3):
            z = analytic_transform(x, time_axis=n)
            self.assertTrue(np.allclose(x - np.mean(x, axis=n, keepdims=True), z.real))

    def test_xarray(self):
        x = xr.DataArray(data=np.random.random((9, 11, 13)))
        for n in range(3):
            z = analytic_transform(x, time_axis=n)
            self.assertTrue(
                np.allclose(x - np.array(np.mean(x, axis=n, keepdims=True)), z.real)
            )


class cartesian_to_rotary_tests(unittest.TestCase):
    def test_size(self):
        u = np.random.rand(99)
        v = np.random.rand(99)
        zp, zn = cartesian_to_rotary(u, v)
        self.assertEqual(np.shape(u), np.shape(zp))
        self.assertEqual(np.shape(u), np.shape(zn))

    def test_array(self):
        u = np.random.random((9))
        v = np.random.random((9))
        zp, zn = cartesian_to_rotary(u, v)
        self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn)))

    def test_ndarray(self):
        u = np.random.rand(99, 10, 101)
        v = np.random.rand(99, 10, 101)
        zp, zn = cartesian_to_rotary(u, v)
        self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn)))

    def test_xarray(self):
        u = xr.DataArray(data=np.random.rand(99))
        v = xr.DataArray(data=np.random.rand(99))
        zp, zn = cartesian_to_rotary(u, v)
        self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn)))


class rotary_to_cartesian_tests(unittest.TestCase):
    def test_size(self):
        zp = np.random.rand(99)
        zn = np.random.rand(99)
        u, v = rotary_to_cartesian(zp, zn)
        self.assertEqual(np.shape(u), np.shape(zp))
        self.assertEqual(np.shape(u), np.shape(zn))

    def test_array(self):
        zp = np.random.random((9))
        zn = np.random.random((9))
        u, v = rotary_to_cartesian(zp, zn)
        self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn)))

    def test_ndarray(self):
        zp = np.random.rand(99, 10, 101)
        zn = np.random.rand(99, 10, 101)
        u, v = rotary_to_cartesian(zp, zn)
        self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn)))

    def test_xarray(self):
        zp = xr.DataArray(data=np.random.rand(99))
        zn = xr.DataArray(data=np.random.rand(99))
        u, v = rotary_to_cartesian(zp, zn)
        self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn)))

    def test_invert_rotary_to_cartesian(self):
        u = np.random.rand(99)
        v = np.random.rand(99)
        zp, zn = cartesian_to_rotary(u, v)
        u_, v_ = rotary_to_cartesian(zp, zn)
        self.assertTrue(np.allclose(u, u_))
        self.assertTrue(np.allclose(v, v_))

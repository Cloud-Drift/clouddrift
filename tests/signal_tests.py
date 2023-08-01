from clouddrift.signal import (
    analytic_transform,
)
import numpy as np
import unittest
import pandas as pd
import xarray as xr

if __name__ == "__main__":
    unittest.main()


def test_analytic_transform_real_odd(self):
    x = np.random.rand(99)
    z = analytic_transform(x)
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z)))


def test_analytic_transform_real_even(self):
    x = np.random.rand(100)
    z = analytic_transform(x)
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z)))


def test_analytic_transform_imag_odd(self):
    z = np.random.rand(99) + 1j * np.random.rand(99)
    zp = analytic_transform(z)
    zn = analytic_transform(np.conj(z))
    self.assertTrue(np.allclose(z - np.mean(z), zp + np.conj(zn)))


def test_analytic_transform_imag_even(self):
    z = np.random.rand(100) + 1j * np.random.rand(100)
    zp = analytic_transform(z)
    zn = analytic_transform(np.conj(z))
    self.assertTrue(np.allclose(z - np.mean(z), zp + np.conj(zn)))


def test_analytic_transform_boundary(self):
    x = np.random.rand(99)
    z1 = analytic_transform(x, boundary="mirror")
    z2 = analytic_transform(x, boundary="zeros")
    z3 = analytic_transform(x, boundary="periodic")
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z1)))
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z2)))
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z3)))


def test_analytic_transform_list(self):
    x = list(np.random.rand(99))
    z = analytic_transform(x)
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z)))


def test_analytic_transform_pandas(self):
    x = pd.Series(data=np.random.rand(99))
    z = analytic_transform(x)
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z)))


def test_analytic_transform_xarray(self):
    x = xr.DataArray(data=np.random.rand(99))
    z = analytic_transform(x)
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z)))

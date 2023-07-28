from clouddrift.analytic import (
    anatrans,
)
import numpy as np
import unittest
import pandas as pd
import xarray as xr

if __name__ == "__main__":
    unittest.main()


def test_anatrans_real_odd(self):
    x = np.random.rand(99)
    z = anatrans(x)
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z)))


def test_anatrans_real_even(self):
    x = np.random.rand(100)
    z = anatrans(x)
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z)))


def test_anatrans_imag_odd(self):
    z = np.random.rand(99) + 1j * np.random.rand(99)
    zp = anatrans(z)
    zn = anatrans(np.conj(z))
    self.assertTrue(np.allclose(z - np.mean(z), zp + np.conj(zn)))


def test_anatrans_imag_even(self):
    z = np.random.rand(100) + 1j * np.random.rand(100)
    zp = anatrans(z)
    zn = anatrans(np.conj(z))
    self.assertTrue(np.allclose(z-np.mean(z), zp + np.conj(zn)))


def test_anatrans_boundary(self):
    x = np.random.rand(99)
    z1 = anatrans(x, boundary="mirror")
    z2 = anatrans(x, boundary="zeros")
    z3 = anatrans(x, boundary="periodic")
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z1)))
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z2)))
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z3)))

def test_anatrans_list(self):
    x = list(np.random.rand(99))
    z = anatrans(x)
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z)))

def test_anatrans_pandas(self):
    x = pd.Series(data=np.random.rand(99))
    z = anatrans(x)
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z)))

def test_anatrans_xarray(self):
    x = xr.DataArray(data=np.random.rand(99))
    z = anatrans(x)
    self.assertTrue(np.allclose(x-np.mean(x), np.real(z)))
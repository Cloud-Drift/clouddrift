from clouddrift.analytic import (
    anatrans,
)
import numpy as np
import unittest

if __name__ == "__main__":
    unittest.main()


def test_anatrans_real_odd(self):
    x = np.random.rand(99)
    x = x - np.mean(x)
    z = anatrans(x)
    self.assertTrue(np.allclose(x, np.real(z)))


def test_anatrans_real_even(self):
    x = np.random.rand(100)
    x = x - np.mean(x)
    z = anatrans(x)
    self.assertTrue(np.allclose(x, np.real(z)))


def test_anatrans_imag_odd(self):
    z = np.random.rand(99) + 1j * np.random.rand(99)
    z = z - np.mean(z)
    zp = anatrans(z)
    zn = anatrans(np.conj(z))
    self.assertTrue(np.allclose(z, zp + np.conj(zn)))


def test_anatrans_imag_even(self):
    z = np.random.rand(100) + 1j * np.random.rand(100)
    z = z - np.mean(z)
    zp = anatrans(z)
    zn = anatrans(np.conj(z))
    self.assertTrue(np.allclose(z, zp + np.conj(zn)))


def test_anatrans_boundary(self):
    x = np.random.rand(99)
    x = x - np.mean(x)
    z1 = anatrans(x, boundary="mirror")
    z2 = anatrans(x, boundary="zeros")
    z3 = anatrans(x, boundary="periodic")
    self.assertTrue(np.allclose(x, np.real(z1)))
    self.assertTrue(np.allclose(x, np.real(z2)))
    self.assertTrue(np.allclose(x, np.real(z3)))

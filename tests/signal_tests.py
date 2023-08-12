from clouddrift.signal import (
    analytic_transform,
    rotary_transform,
)
import numpy as np
import unittest
import xarray as xr

if __name__ == "__main__":
    unittest.main()


def test_analytic_transform_real_odd(self):
    x = np.random.rand(99)
    z = analytic_transform(x)
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z),atol=0.1,rtol=0))


def test_analytic_transform_real_even(self):
    x = np.random.rand(100)
    z = analytic_transform(x)
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z),atol=0.1,rtol=0))


def test_analytic_transform_imag_odd(self):
    z = np.random.rand(99) + 1j * np.random.rand(99)
    zp = analytic_transform(z)
    zn = analytic_transform(np.conj(z))
    self.assertTrue(np.allclose(z - np.mean(z,keepdims=True), zp + np.conj(zn),atol=0.1,rtol=0))


def test_analytic_transform_imag_even(self):
    z = np.random.rand(100) + 1j * np.random.rand(100)
    zp = analytic_transform(z)
    zn = analytic_transform(np.conj(z))
    self.assertTrue(np.allclose(z - np.mean(z,keepdims=True), zp + np.conj(zn),atol=0.1,rtol=0))


def test_analytic_transform_boundary(self):
    x = np.random.rand(99)
    z1 = analytic_transform(x, boundary="mirror")
    z2 = analytic_transform(x, boundary="zeros")
    z3 = analytic_transform(x, boundary="periodic")
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z1),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z2),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - np.mean(x), np.real(z3),atol=0.1,rtol=0))


def test_analytic_transform_ndarray(self):
    x = np.random.rand(99,10,101)
    z0 = analytic_transform(x,time_axis=0)
    z1 = analytic_transform(x,time_axis=1)
    z2 = analytic_transform(x,time_axis=2)
    z3 = analytic_transform(x,time_axis=-1)
    self.assertTrue(np.allclose(x - np.mean(x,axis=0,keepdims=True), np.real(z0),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - np.mean(x,axis=1,keepdims=True), np.real(z1),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - np.mean(x,axis=2,keepdims=True), np.real(z2),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - np.mean(x,axis=-1,keepdims=True), np.real(z3),atol=0.1,rtol=0))


def test_analytic_transform_xarray(self):
    x = xr.DataArray(data=np.random.rand(99,10,101))
    z0 = analytic_transform(x,time_axis=0)
    z1 = analytic_transform(x,time_axis=1)
    z2 = analytic_transform(x,time_axis=2)
    z3 = analytic_transform(x,time_axis=-1)
    self.assertTrue(np.allclose(x - x.mean(axis=0), np.real(z0),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - x.mean(axis=1), np.real(z1),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - x.mean(axis=2), np.real(z2),atol=0.1,rtol=0))
    self.assertTrue(np.allclose(x - x.mean(axis=-1), np.real(z3),atol=0.1,rtol=0))


def test_rotary_transform_array(self):
    u = np.random.rand(99)
    v = np.random.rand(99)
    zp, zn = rotary_transform(u, v)
    self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn),atol=0.1,rtol=0))

def test_rotary_transform_ndarray(self):
    u = np.random.rand(99,10,101)
    v = np.random.rand(99,10,101)
    zp, zn = rotary_transform(u, v)
    self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn),atol=0.1,rtol=0))

def test_rotary_transform_xarray(self):
    u = xr.DataArray(data=np.random.rand(99))
    v = xr.DataArray(data=np.random.rand(99))
    zp, zn = rotary_transform(u, v)
    self.assertTrue(np.allclose(u + 1j * v, zp + np.conj(zn),atol=0.1,rtol=0))

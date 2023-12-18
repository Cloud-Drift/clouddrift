from clouddrift import datasets
from clouddrift.ragged import apply_ragged, subset
import numpy as np
import unittest


if __name__ == "__main__":
    unittest.main()


class datasets_tests(unittest.TestCase):
    def test_gdp1h(self):
        ds = datasets.gdp1h()
        self.assertTrue(ds)

    def test_gdp6h(self):
        ds = datasets.gdp6h()
        self.assertTrue(ds)

    def test_glad(self):
        ds = datasets.glad()
        self.assertTrue(ds)

    def test_glad_dims_coords(self):
        ds = datasets.glad()
        self.assertTrue(len(ds.sizes) == 2)
        self.assertTrue("obs" in ds.dims)
        self.assertTrue("traj" in ds.dims)
        self.assertTrue(len(ds.coords) == 2)
        self.assertTrue("time" in ds.coords)
        self.assertTrue("id" in ds.coords)

    def test_glad_subset_and_apply_ragged_work(self):
        ds = datasets.glad()
        ds_sub = subset(ds, {"id": ["CARTHE_001", "CARTHE_002"]}, id_var_name="id")
        self.assertTrue(ds_sub)
        mean_lon = apply_ragged(np.mean, [ds_sub.longitude], ds_sub.rowsize)
        self.assertTrue(mean_lon.size == 2)

    def test_spotters_opens(self):
        ds = datasets.spotters()
        self.assertTrue(ds)

    def test_subsurface_floats_opens(self):
        ds = datasets.subsurface_floats()
        self.assertTrue(ds)

    def test_andro_opens(self):
        ds = datasets.andro()
        self.assertTrue(ds)

    def test_yomaha_opens(self):
        ds = datasets.yomaha()
        self.assertTrue(ds)

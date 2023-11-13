from clouddrift import datasets
from clouddrift.ragged import apply_ragged, subset
import numpy as np
import unittest


if __name__ == "__main__":
    unittest.main()


class datasets_tests(unittest.TestCase):
    def test_gdp1h_opens(self):
        ds = datasets.gdp1h()
        self.assertTrue(ds)

    def test_gdp6h_opens(self):
        ds = datasets.gdp6h()
        self.assertTrue(ds)

    def test_glad_opens(self):
        ds = datasets.glad()
        self.assertTrue(ds)

    def test_glad_subset_and_apply_ragged_work(self):
        ds = datasets.glad()
        ds_sub = subset(ds, {"traj": ["CARTHE_001", "CARTHE_002"]}, id_var_name="traj")
        self.assertTrue(ds_sub)
        mean_lon = apply_ragged(np.mean, [ds_sub.longitude], ds_sub.rowsize)
        self.assertTrue(mean_lon.size == 2)

from clouddrift import pairs
import numpy as np
import pandas as pd
import unittest
import xarray as xr


if __name__ == "__main__":
    unittest.main()


class pairs_time_overlap_tests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = [0, 1, 2, 3]
        self.b = [2, 3, 4, 5]

    def test_list(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.b, 0.5)
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_ndarray(self):
        mask1, mask2 = pairs.pair_time_overlap(np.array(self.a), np.array(self.b), 0.5)
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_xarray(self):
        mask1, mask2 = pairs.pair_time_overlap(
            xr.DataArray(data=self.a), xr.DataArray(data=self.b), 0.5
        )
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_pandas(self):
        mask1, mask2 = pairs.pair_time_overlap(
            pd.Series(self.a), pd.Series(self.b), 0.5
        )
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_tolerance(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.b, 1.5)
        self.assertTrue(np.all(mask1 == [False, True, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, True, False]))

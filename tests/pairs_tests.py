from clouddrift import datasets, pairs, ragged, sphere
import numpy as np
import pandas as pd
import unittest
import xarray as xr


if __name__ == "__main__":
    unittest.main()


class pairs_chance_pair_tests(unittest.TestCase):
    def setUp(self) -> None:
        ds = datasets.glad()
        self.lon1 = ragged.unpack(ds["longitude"], ds["rowsize"], rows=0).pop()
        self.lat1 = ragged.unpack(ds["latitude"], ds["rowsize"], rows=0).pop()
        self.time1 = ragged.unpack(ds["time"], ds["rowsize"], rows=0).pop()
        self.lon2 = ragged.unpack(ds["longitude"], ds["rowsize"], rows=1).pop()
        self.lat2 = ragged.unpack(ds["latitude"], ds["rowsize"], rows=1).pop()
        self.time2 = ragged.unpack(ds["time"], ds["rowsize"], rows=1).pop()

    def test_chance_pair(self):
        i1, i2 = pairs.chance_pair(
            self.lon1,
            self.lat1,
            self.lon2,
            self.lat2,
            self.time1,
            self.time2,
            space_tolerance=5000,
            time_tolerance=np.timedelta64(1),
        )
        space_distance = sphere.distance(
            self.lon1[i1], self.lat1[i1], self.lon2[i2], self.lat2[i2]
        )
        self.assertTrue(np.all(space_distance < 5000))


class pairs_bounding_box_overlap_tests(unittest.TestCase):
    def setUp(self) -> None:
        self.lon1 = np.arange(4)
        self.lat1 = np.arange(4)
        self.lon2 = np.arange(2, 6)
        self.lat2 = np.arange(2, 6)
        self.lon3 = np.arange(2, 7)
        self.lat3 = np.arange(2, 7)

    def test_list(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            self.lon1, self.lat1, self.lon2, self.lat2, 0.5
        )
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_ndarray(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            np.array(self.lon1),
            np.array(self.lat1),
            np.array(self.lon2),
            np.array(self.lat2),
            0.5,
        )
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_xarray(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            xr.DataArray(data=self.lon1),
            xr.DataArray(data=self.lat1),
            xr.DataArray(data=self.lon2),
            xr.DataArray(data=self.lat2),
            0.5,
        )
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_pandas(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            pd.Series(self.lon1),
            pd.Series(self.lat1),
            pd.Series(self.lon2),
            pd.Series(self.lat2),
            0.5,
        )
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False]))

    def test_tolerance(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            self.lon1, self.lat1, self.lon2, self.lat2, 1.5
        )
        self.assertTrue(np.all(mask1 == [False, True, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, True, False]))

    def test_different_length_inputs(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            self.lon1, self.lat1, self.lon3, self.lat3, 0.5
        )
        self.assertTrue(np.all(mask1 == [False, False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False, False, False]))

    def test_no_overlap(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            np.array([0, 0]),
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([1, 1]),
            0.5,
        )
        self.assertTrue(np.all(mask1 == [False, False]))
        self.assertTrue(np.all(mask2 == [False, False]))


class pairs_time_overlap_tests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = [0, 1, 2]
        self.b = [2, 3, 4]
        self.c = [2, 3, 4, 5]

    def test_list(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.b, 0.5)
        self.assertTrue(np.all(mask1 == [False, False, True]))
        self.assertTrue(np.all(mask2 == [True, False, False]))

    def test_ndarray(self):
        mask1, mask2 = pairs.pair_time_overlap(np.array(self.a), np.array(self.b), 0.5)
        self.assertTrue(np.all(mask1 == [False, False, True]))
        self.assertTrue(np.all(mask2 == [True, False, False]))

    def test_xarray(self):
        mask1, mask2 = pairs.pair_time_overlap(
            xr.DataArray(data=self.a), xr.DataArray(data=self.b), 0.5
        )
        self.assertTrue(np.all(mask1 == [False, False, True]))
        self.assertTrue(np.all(mask2 == [True, False, False]))

    def test_pandas(self):
        mask1, mask2 = pairs.pair_time_overlap(
            pd.Series(self.a), pd.Series(self.b), 0.5
        )
        self.assertTrue(np.all(mask1 == [False, False, True]))
        self.assertTrue(np.all(mask2 == [True, False, False]))

    def test_tolerance(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.b, 1.5)
        self.assertTrue(np.all(mask1 == [False, True, True]))
        self.assertTrue(np.all(mask2 == [True, True, False]))

    def test_different_length_inputs(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.c, 0.5)
        self.assertTrue(np.all(mask1 == [False, False, True]))
        self.assertTrue(np.all(mask2 == [True, False, False, False]))

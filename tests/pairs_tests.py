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
        space_distance = 6000
        time_distance = np.timedelta64(0)
        lon1, lat1, lon2, lat2, time1, time2 = pairs.chance_pair(
            self.lon1,
            self.lat1,
            self.lon2,
            self.lat2,
            self.time1,
            self.time2,
            space_distance,
            time_distance,
        )
        self.assertTrue(
            np.all(sphere.distance(lon1, lat1, lon2, lat2) <= space_distance)
        )
        self.assertTrue(np.all(time1 == time2))


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
            self.lon1,
            self.lat1,
            self.lon2,
            self.lat2,
        )
        self.assertTrue(np.all(mask1 == [2, 3]))
        self.assertTrue(np.all(mask2 == [0, 1]))

    def test_ndarray(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            np.array(self.lon1),
            np.array(self.lat1),
            np.array(self.lon2),
            np.array(self.lat2),
        )
        self.assertTrue(np.all(mask1 == [2, 3]))
        self.assertTrue(np.all(mask2 == [0, 1]))

    def test_xarray(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            xr.DataArray(data=self.lon1),
            xr.DataArray(data=self.lat1),
            xr.DataArray(data=self.lon2),
            xr.DataArray(data=self.lat2),
        )
        self.assertTrue(np.all(mask1 == [2, 3]))
        self.assertTrue(np.all(mask2 == [0, 1]))

    def test_pandas(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            pd.Series(self.lon1),
            pd.Series(self.lat1),
            pd.Series(self.lon2),
            pd.Series(self.lat2),
        )
        self.assertTrue(np.all(mask1 == [2, 3]))
        self.assertTrue(np.all(mask2 == [0, 1]))

    def test_distance(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            self.lon1, self.lat1, self.lon2, self.lat2, 1
        )
        self.assertTrue(np.all(mask1 == [1, 2, 3]))
        self.assertTrue(np.all(mask2 == [0, 1, 2]))

    def test_different_length_inputs(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            self.lon1,
            self.lat1,
            self.lon3,
            self.lat3,
        )
        self.assertTrue(np.all(mask1 == [2, 3]))
        self.assertTrue(np.all(mask2 == [0, 1]))

    def test_no_overlap(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            np.array([0, 0]),
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([1, 1]),
        )
        self.assertTrue(np.all(mask1 == []))
        self.assertTrue(np.all(mask2 == []))


class pairs_time_overlap_tests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = [0, 1, 2]
        self.b = [2, 3, 4]
        self.c = [2, 3, 4, 5]

    def test_list(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.b)
        self.assertTrue(np.all(mask1 == [2]))
        self.assertTrue(np.all(mask2 == [0]))

    def test_ndarray(self):
        mask1, mask2 = pairs.pair_time_overlap(np.array(self.a), np.array(self.b))
        self.assertTrue(np.all(mask1 == [2]))
        self.assertTrue(np.all(mask2 == [0]))

    def test_xarray(self):
        mask1, mask2 = pairs.pair_time_overlap(
            xr.DataArray(data=self.a),
            xr.DataArray(data=self.b),
        )
        self.assertTrue(np.all(mask1 == [2]))
        self.assertTrue(np.all(mask2 == [0]))

    def test_pandas(self):
        mask1, mask2 = pairs.pair_time_overlap(
            pd.Series(self.a),
            pd.Series(self.b),
        )
        self.assertTrue(np.all(mask1 == [2]))
        self.assertTrue(np.all(mask2 == [0]))

    def test_distance(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.b, 1)
        self.assertTrue(np.all(mask1 == [1, 2]))
        self.assertTrue(np.all(mask2 == [0, 1]))

    def test_different_length_inputs(self):
        mask1, mask2 = pairs.pair_time_overlap(self.a, self.c, 1)
        self.assertTrue(np.all(mask1 == [1, 2]))
        self.assertTrue(np.all(mask2 == [0, 1]))

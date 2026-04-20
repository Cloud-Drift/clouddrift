import unittest

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift import datasets, pairs, ragged, sphere

if __name__ == "__main__":
    unittest.main()


class pairs_chance_pairs_from_ragged_tests(unittest.TestCase):
    def setUp(self) -> None:
        num_trajectories = 10
        ids = ["CARTHE_%3.3i" % (i + 1) for i in range(num_trajectories)]
        ds = ragged.subset(
            datasets.glad(), {"id": ids}, id_var_name="id", row_dim_name="traj"
        )
        self.lon = ds["longitude"]
        self.lat = ds["latitude"]
        self.time = ds["time"]
        self.rowsize = ds["rowsize"]

    def test_chance_pairs_from_ragged(self):
        space_distance = 10000
        time_distance = np.timedelta64(0)
        chance_pairs = pairs.chance_pairs_from_ragged(
            self.lon,
            self.lat,
            self.rowsize,
            space_distance,
            self.time,
            time_distance,
        )
        for pair in chance_pairs:
            rows = pair[0]
            i1, i2 = pair[1]
            lon1 = ragged.unpack(self.lon, self.rowsize, rows=rows[0]).pop()
            lat1 = ragged.unpack(self.lat, self.rowsize, rows=rows[0]).pop()
            time1 = ragged.unpack(self.time, self.rowsize, rows=rows[0]).pop()
            lon2 = ragged.unpack(self.lon, self.rowsize, rows=rows[1]).pop()
            lat2 = ragged.unpack(self.lat, self.rowsize, rows=rows[1]).pop()
            time2 = ragged.unpack(self.time, self.rowsize, rows=rows[1]).pop()
            self.assertTrue(
                np.all(
                    sphere.distance(lon1[i1], lat1[i1], lon2[i2], lat2[i2])
                    <= space_distance
                )
            )
            self.assertTrue(np.all(time1[i1] == time2[i2]))


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
        i1, i2 = pairs.chance_pair(
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
            np.all(
                sphere.distance(
                    self.lon1[i1], self.lat1[i1], self.lon2[i2], self.lat2[i2]
                )
                <= space_distance
            )
        )
        self.assertTrue(np.all(self.time1[i1] == self.time2[i2]))


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

    def test_antimeridian(self):
        mask1, mask2 = pairs.pair_bounding_box_overlap(
            [179, -179], [0, 2], [0, 2], [0, 2]
        )
        self.assertTrue(mask1.size == mask2.size == 0)


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

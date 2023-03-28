from clouddrift.analysis import segment, velocity_from_position, apply_ragged, subset
from clouddrift.haversine import EARTH_RADIUS_METERS
from clouddrift.dataformat import RaggedArray
import unittest
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta


if __name__ == "__main__":
    unittest.main()


class segment_tests(unittest.TestCase):
    def test_segment(self):
        x = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
        tol = 0.5
        self.assertTrue(type(segment(x, tol)) is np.ndarray)
        self.assertTrue(np.all(segment(x, tol) == np.array([1, 3, 2, 4, 1])))
        self.assertTrue(np.all(segment(np.array(x), tol) == np.array([1, 3, 2, 4, 1])))
        self.assertTrue(
            np.all(segment(xr.DataArray(data=x), tol) == np.array([1, 3, 2, 4, 1]))
        )

    def test_segment_zero_tolerance(self):
        x = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        tol = 0
        self.assertIsNone(
            np.testing.assert_equal(segment(x, tol), np.array([1, 2, 3, 4]))
        )

    def test_segment_negative_tolerance(self):
        x = [0, 1, 1, 1, 2, 0, 3, 3, 3, 4]
        tol = -1
        self.assertTrue(np.all(segment(x, tol) == np.array([5, 5])))

    def test_segment_rowsize(self):
        x = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
        tol = 0.5
        rowsize = [6, 5]
        segment_sizes = segment(x, tol, rowsize)
        self.assertTrue(type(segment_sizes) is np.ndarray)
        self.assertTrue(np.all(segment_sizes == np.array([1, 3, 2, 4, 1])))

    def test_segment_positive_and_negative_tolerance(self):
        x = [1, 1, 2, 2, 1, 1, 2, 2]
        segment_sizes = segment(x, 0.5, rowsize=segment(x, -0.5))
        self.assertTrue(np.all(segment_sizes == np.array([2, 2, 2, 2])))

    def test_segment_rowsize_raises(self):
        x = [0, 1, 2, 3]
        tol = 0.5
        rowsize = [1, 2]  # rowsize is too short
        with self.assertRaises(ValueError):
            segment(x, tol, rowsize)

    def test_segments_datetime(self):
        x = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 2, 1),
            datetime(2023, 2, 2),
        ]
        for tol in [pd.Timedelta("1 day"), timedelta(days=1), np.timedelta64(1, "D")]:
            self.assertIsNone(
                np.testing.assert_equal(segment(x, tol), np.array([3, 2]))
            )

    def test_segments_numpy(self):
        x = np.array(
            [
                np.datetime64("2023-01-01"),
                np.datetime64("2023-01-02"),
                np.datetime64("2023-01-03"),
                np.datetime64("2023-02-01"),
                np.datetime64("2023-02-02"),
            ]
        )
        for tol in [pd.Timedelta("1 day"), timedelta(days=1), np.timedelta64(1, "D")]:
            self.assertIsNone(
                np.testing.assert_equal(segment(x, tol), np.array([3, 2]))
            )

    def test_segments_pandas(self):
        x = pd.to_datetime(["1/1/2023", "1/2/2023", "1/3/2023", "2/1/2023", "2/2/2023"])
        for tol in [pd.Timedelta("1 day"), timedelta(days=1), np.timedelta64(1, "D")]:
            self.assertIsNone(
                np.testing.assert_equal(segment(x, tol), np.array([3, 2]))
            )


class velocity_from_position_tests(unittest.TestCase):
    def setUp(self):
        self.INPUT_SIZE = 100
        self.lon = np.rad2deg(np.linspace(-np.pi, np.pi, self.INPUT_SIZE))
        self.lat = np.zeros(self.lon.shape)
        self.time = np.linspace(0, 1e7, self.INPUT_SIZE)
        self.uf, self.vf = velocity_from_position(self.lon, self.lat, self.time)
        self.ub, self.vb = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="backward"
        )
        self.uc, self.vc = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="centered"
        )

    def test_result_has_same_size_as_input(self):
        self.assertTrue(np.all(self.uf.shape == self.vf.shape == self.lon.shape))
        self.assertTrue(np.all(self.ub.shape == self.vb.shape == self.lon.shape))
        self.assertTrue(np.all(self.uc.shape == self.vc.shape == self.lon.shape))

    def test_schemes_are_self_consistent(self):
        self.assertTrue(np.all(self.uf[:-1] == self.ub[1:]))
        self.assertTrue(
            np.all(np.isclose((self.uf[1:-1] + self.ub[1:-1]) / 2, self.uc[1:-1]))
        )
        self.assertTrue(self.uc[0] == self.uf[0])
        self.assertTrue(self.uc[-1] == self.ub[-1])

    def test_result_value(self):
        u_expected = 2 * np.pi * EARTH_RADIUS_METERS / 1e7
        self.assertTrue(np.all(np.isclose(self.uf, u_expected)))
        self.assertTrue(np.all(np.isclose(self.ub, u_expected)))
        self.assertTrue(np.all(np.isclose(self.uc, u_expected)))

    def test_works_with_dataarray(self):
        lon = xr.DataArray(data=self.lon, coords={"time": self.time})
        lat = xr.DataArray(data=self.lat, coords={"time": self.time})
        time = xr.DataArray(data=self.time, coords={"time": self.time})
        uf, vf = velocity_from_position(lon, lat, time)
        self.assertTrue(np.all(uf == self.uf))
        self.assertTrue(np.all(vf == self.vf))

    def test_works_with_2d_array(self):
        lon = np.reshape(np.tile(self.lon, 4), (4, self.lon.size))
        lat = np.reshape(np.tile(self.lat, 4), (4, self.lat.size))
        time = np.reshape(np.tile(self.time, 4), (4, self.time.size))
        expected_uf = np.reshape(np.tile(self.uf, 4), (4, self.uf.size))
        expected_vf = np.reshape(np.tile(self.vf, 4), (4, self.vf.size))
        uf, vf = velocity_from_position(lon, lat, time)
        self.assertTrue(np.all(uf == expected_uf))
        self.assertTrue(np.all(vf == expected_vf))
        self.assertTrue(np.all(uf.shape == expected_uf.shape))
        self.assertTrue(np.all(vf.shape == expected_vf.shape))

    def test_works_with_3d_array(self):
        lon = np.reshape(np.tile(self.lon, 4), (2, 2, self.lon.size))
        lat = np.reshape(np.tile(self.lat, 4), (2, 2, self.lat.size))
        time = np.reshape(np.tile(self.time, 4), (2, 2, self.time.size))
        expected_uf = np.reshape(np.tile(self.uf, 4), (2, 2, self.uf.size))
        expected_vf = np.reshape(np.tile(self.vf, 4), (2, 2, self.vf.size))
        uf, vf = velocity_from_position(lon, lat, time)
        self.assertTrue(np.all(uf == expected_uf))
        self.assertTrue(np.all(vf == expected_vf))
        self.assertTrue(np.all(uf.shape == expected_uf.shape))
        self.assertTrue(np.all(vf.shape == expected_vf.shape))

    def test_time_axis(self):
        lon = np.transpose(
            np.reshape(np.tile(self.lon, 4), (2, 2, self.lon.size)), (0, 2, 1)
        )
        lat = np.transpose(
            np.reshape(np.tile(self.lat, 4), (2, 2, self.lat.size)), (0, 2, 1)
        )
        time = np.transpose(
            np.reshape(np.tile(self.time, 4), (2, 2, self.time.size)), (0, 2, 1)
        )
        expected_uf = np.transpose(
            np.reshape(np.tile(self.uf, 4), (2, 2, self.uf.size)), (0, 2, 1)
        )
        expected_vf = np.transpose(
            np.reshape(np.tile(self.vf, 4), (2, 2, self.vf.size)), (0, 2, 1)
        )
        uf, vf = velocity_from_position(lon, lat, time, time_axis=1)
        self.assertTrue(np.all(uf == expected_uf))
        self.assertTrue(np.all(vf == expected_vf))
        self.assertTrue(np.all(uf.shape == expected_uf.shape))
        self.assertTrue(np.all(vf.shape == expected_vf.shape))


class apply_ragged_tests(unittest.TestCase):
    def setUp(self):
        self.rowsize = [2, 3, 4]
        self.x = np.array([1, 2, 10, 12, 14, 30, 33, 36, 39])
        self.y = np.arange(0, len(self.x))
        self.t = np.array([1, 2, 1, 2, 3, 1, 2, 3, 4])

    def test_simple(self):
        y = apply_ragged(lambda x: x**2, np.array([1, 2, 3, 4]), [2, 2])
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_simple_dataarray(self):
        y = apply_ragged(
            lambda x: x**2,
            xr.DataArray(data=[1, 2, 3, 4], coords={"obs": [1, 2, 3, 4]}),
            [2, 2],
        )
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_simple_with_args(self):
        y = apply_ragged(lambda x, p: x**p, np.array([1, 2, 3, 4]), [2, 2], 2)
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_simple_with_kwargs(self):
        y = apply_ragged(lambda x, p: x**p, np.array([1, 2, 3, 4]), [2, 2], p=2)
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_velocity_ndarray(self):
        u, v = apply_ragged(
            velocity_from_position,
            [self.x, self.y, self.t],
            self.rowsize,
            coord_system="cartesian",
        )
        self.assertIsNone(
            np.testing.assert_allclose(u, [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        )
        self.assertIsNone(
            np.testing.assert_allclose(v, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )

    def test_velocity_dataarray(self):
        u, v = apply_ragged(
            velocity_from_position,
            [
                xr.DataArray(data=self.x),
                xr.DataArray(data=self.y),
                xr.DataArray(data=self.t),
            ],
            xr.DataArray(data=self.rowsize),
            coord_system="cartesian",
        )
        self.assertIsNone(
            np.testing.assert_allclose(u, [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        )
        self.assertIsNone(
            np.testing.assert_allclose(v, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )

    def test_bad_rowsize_raises(self):
        with self.assertRaises(ValueError):
            y = apply_ragged(lambda x: x**2, np.array([1, 2, 3, 4]), [2])


class subset_tests(unittest.TestCase):
    def setUp(self):
        """
        Create ragged array and output netCDF and Parquet file
        """
        drifter_id = [1, 2, 3]
        rowsize = [5, 4, 2]
        longitude = [[-121, -111, 51, 61, 71], [12, 22, 32, 42], [103, 113]]
        latitude = [[-90, -45, 45, 90, 0], [10, 20, 30, 40], [10, 20]]
        t = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [4, 5]]
        ids = [[1, 1, 1, 1, 1], [2, 2, 2, 2], [3, 3]]
        test = [
            [True, True, True, False, False],
            [True, False, False, False],
            [False, False],
        ]
        nb_obs = np.sum(rowsize)
        nb_traj = len(drifter_id)
        attrs_global = {
            "title": "test trajectories",
            "history": "version xyz",
        }
        variables_coords = ["ids", "time", "lon", "lat"]

        coords = {"lon": longitude, "lat": latitude, "ids": ids, "time": t}
        metadata = {"ID": drifter_id, "rowsize": rowsize}
        data = {"test": test}

        # append xr.Dataset to a list
        list_ds = []
        for i in range(0, len(rowsize)):
            xr_coords = {}
            for var in coords.keys():
                xr_coords[var] = (
                    ["obs"],
                    coords[var][i],
                    {"long_name": f"variable {var}", "units": "-"},
                )

            xr_data = {}
            for var in metadata.keys():
                xr_data[var] = (
                    ["traj"],
                    [metadata[var][i]],
                    {"long_name": f"variable {var}", "units": "-"},
                )

            for var in data.keys():
                xr_data[var] = (
                    ["obs"],
                    data[var][i],
                    {"long_name": f"variable {var}", "units": "-"},
                )

            list_ds.append(
                xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=attrs_global)
            )

        # create test ragged array
        ra = RaggedArray.from_files(
            [0, 1, 2],
            lambda i: list_ds[i],
            variables_coords,
            ["ID", "rowsize"],
            ["test"],
        )

        self.ds = ra.to_xarray()

    def test_equal(self):
        ds_sub = subset(self.ds, {"test": True})
        self.assertEqual(len(ds_sub.ID), 2)

    def test_select(self):
        ds_sub = subset(self.ds, {"ID": [1, 2]})
        self.assertTrue(all(ds_sub.ID == [1, 2]))
        self.assertEqual(len(ds_sub.ID), 2)

    def test_range(self):
        # positive
        ds_sub = subset(self.ds, {"lon": (0, 180)})
        traj_idx = np.insert(np.cumsum(ds_sub["rowsize"].values), 0, 0)
        self.assertTrue(
            all(ds_sub.lon[slice(traj_idx[0], traj_idx[1])] == [51, 61, 71])
        )
        self.assertTrue(
            all(ds_sub.lon[slice(traj_idx[1], traj_idx[2])] == [12, 22, 32, 42])
        )
        self.assertTrue(all(ds_sub.lon[slice(traj_idx[2], traj_idx[3])] == [103, 113]))

        # negative range
        ds_sub = subset(self.ds, {"lon": (-180, 0)})
        traj_idx = np.insert(np.cumsum(ds_sub["rowsize"].values), 0, 0)
        self.assertEqual(len(ds_sub.ID), 1)
        self.assertEqual(ds_sub.ID[0], 1)
        self.assertTrue(all(ds_sub.lon == [-121, -111]))

        # both
        ds_sub = subset(self.ds, {"lon": (-30, 30)})
        traj_idx = np.insert(np.cumsum(ds_sub["rowsize"].values), 0, 0)
        self.assertEqual(len(ds_sub.ID), 1)
        self.assertEqual(ds_sub.ID[0], 2)
        self.assertTrue(all(ds_sub.lon[slice(traj_idx[0], traj_idx[1])] == ([12, 22])))

    def test_combine(self):
        ds_sub = subset(
            self.ds, {"ID": [1, 2], "lat": (-90, 20), "lon": (-180, 25), "test": True}
        )
        self.assertTrue(all(ds_sub.ID == [1, 2]))
        self.assertTrue(all(ds_sub.lon == [-121, -111, 12]))
        self.assertTrue(all(ds_sub.lat == [-90, -45, 10]))

    def test_empty(self):
        ds_sub = subset(self.ds, {"ID": 3, "lon": (-180, 0)})
        self.assertTrue(ds_sub.dims == {})

    def test_unknown_var(self):
        with self.assertRaises(ValueError):
            subset(self.ds, {"a": 10})

        with self.assertRaises(ValueError):
            subset(self.ds, {"lon": (0, 180), "a": (0, 10)})

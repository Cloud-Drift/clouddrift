import unittest
from concurrent import futures
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift.kinematics import velocity_from_position
from clouddrift.ragged import (
    apply_ragged,
    chunk,
    prune,
    ragged_to_regular,
    regular_to_ragged,
    segment,
    subset,
    unpack,
)
from clouddrift.raggedarray import RaggedArray

if __name__ == "__main__":
    unittest.main()


def sample_ragged_array() -> RaggedArray:
    drifter_id = [1, 3, 2]
    longitude = [[-121, -111, 51, 61, 71], [103, 113], [12, 22, 32, 42]]
    latitude = [[-90, -45, 45, 90, 0], [10, 20], [10, 20, 30, 40]]
    t = [[1, 2, 3, 4, 5], [4, 5], [2, 3, 4, 5]]
    test = [
        [True, True, True, False, False],
        [False, False],
        [True, False, False, False],
    ]
    rowsize = [len(x) for x in longitude]
    attrs_global = {
        "title": "test trajectories",
        "history": "version xyz",
    }
    coords: dict[str, list] = {"id": drifter_id, "time": t}
    metadata = {"rowsize": rowsize}
    data: dict[str, list] = {"test": test, "lat": latitude, "lon": longitude}

    # append xr.Dataset to a list
    list_ds = []
    for i in range(0, len(rowsize)):
        xr_coords = {}
        xr_coords["id"] = (
            ["rows"],
            [coords["id"][i]],
            {"long_name": "variable id", "units": "-"},
        )
        xr_coords["time"] = (
            ["obs"],
            coords["time"][i],
            {"long_name": "variable time", "units": "-"},
        )

        xr_data: dict[str, Any] = {}
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

    ra = RaggedArray.from_files(
        [0, 1, 2],
        lambda i: list_ds[i],
        ["id", "time"],
        name_meta=["rowsize"],
        name_data=["test", "lat", "lon"],
        name_dims={"rows": "rows", "obs": "obs"}
    )

    return ra


class chunk_tests(unittest.TestCase):
    def test_chunk(self):
        # Simple chunk without trimming
        self.assertTrue(np.all(chunk([1, 2, 3, 4], 2) == np.array([[1, 2], [3, 4]])))

        # Simple chunk with trimming
        self.assertTrue(np.all(chunk([1, 2, 3, 4, 5], 2) == np.array([[1, 2], [3, 4]])))

        # Simple chunk with trimming skipping the first point
        self.assertTrue(
            np.all(chunk([1, 2, 3, 4, 5], 2, align="end") == np.array([[2, 3], [4, 5]]))
        )

        # Simple chunk with trimming skipping the first point
        self.assertTrue(
            np.all(
                chunk([1, 2, 3, 4, 5, 6, 7, 8], 3, align="end")
                == np.array([[3, 4, 5], [6, 7, 8]])
            )
        )

        # Simple chunk with trimming with middle alignment
        self.assertTrue(
            np.all(
                chunk([1, 2, 3, 4, 5, 6, 7, 8], 3, align="middle")
                == np.array([[2, 3, 4], [5, 6, 7]])
            )
        )

        # Simple chunk with align to the end with with overlap
        self.assertTrue(
            np.all(
                chunk([1, 2, 3, 4, 5, 6, 7, 8], 3, 1, align="end")
                == np.array([[2, 3, 4], [4, 5, 6], [6, 7, 8]])
            )
        )

        # Simple chunk with trimming skipping the first point with overlap
        self.assertTrue(
            np.all(
                chunk(np.arange(1, 12), 4, align="middle")
                == np.array([[2, 3, 4, 5], [6, 7, 8, 9]])
            )
        )

        # When length == 1, result is a transpose of the input
        self.assertTrue(np.all(chunk([1, 2, 3, 4], 1) == np.array([[1, 2, 3, 4]]).T))

        # When length > len(x), result is an empty 2-d array
        self.assertTrue(chunk([1], 2).shape == (0, 2))

        # When length == 0, the function raises a ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            chunk([1], 0)

        # When length < 0, the function raises a ValueError
        with self.assertRaises(ValueError):
            chunk([1], -1)

        # When align is assigned a wrong value, the function raises a ValueError
        with self.assertRaises(ValueError):
            chunk([1], 1, align="wrong")

    def test_chunk_overlap(self):
        # Simple chunk with overlap
        self.assertTrue(
            np.all(
                chunk([1, 2, 3, 4, 5], 2, 1)
                == np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
            )
        )

        # Overlap larger than length raises ValueError
        with self.assertRaises(ValueError):
            chunk([1, 2, 3, 4, 5], 2, 3)

        # Negative overlap offsets chunks
        self.assertTrue(
            np.all(chunk([1, 2, 3, 4, 5], 2, -1) == np.array([[1, 2], [4, 5]]))
        )

    def test_chunk_rowsize(self):
        # Simple chunk with rowsize
        self.assertTrue(
            np.all(
                apply_ragged(chunk, np.array([1, 2, 3, 4, 5, 6]), [2, 3, 1], 2)
                == np.array([[1, 2], [3, 4]])
            )
        )

        # rowsize with overlap
        self.assertTrue(
            np.all(
                apply_ragged(
                    chunk, np.array([1, 2, 3, 4, 5, 6]), [2, 3, 1], 2, overlap=1
                )
                == np.array([[1, 2], [3, 4], [4, 5]])
            )
        )

    def test_chunk_array_like(self):
        # chunk works with array-like objects
        self.assertTrue(
            np.all(chunk(np.array([1, 2, 3, 4]), 2) == np.array([[1, 2], [3, 4]]))
        )
        self.assertTrue(
            np.all(
                chunk(xr.DataArray(data=[1, 2, 3, 4]), 2) == np.array([[1, 2], [3, 4]])
            )
        )
        self.assertTrue(
            np.all(chunk(pd.Series(data=[1, 2, 3, 4]), 2) == np.array([[1, 2], [3, 4]]))
        )


class prune_tests(unittest.TestCase):
    def test_prune(self):
        x = [1, 2, 3, 1, 2, 1, 2, 3, 4]
        rowsize = [3, 2, 4]
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, rowsize_new = prune(data, rowsize, minimum)
            self.assertTrue(type(x_new) is np.ndarray)
            self.assertTrue(type(rowsize_new) is np.ndarray)
            np.testing.assert_equal(x_new, [1, 2, 3, 1, 2, 3, 4])
            np.testing.assert_equal(rowsize_new, [3, 4])

    def test_prune_all_longer(self):
        x = [1, 2, 3, 1, 2, 1, 2, 3, 4]
        rowsize = [3, 2, 4]
        minimum = 1

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, rowsize_new = prune(data, rowsize, minimum)
            np.testing.assert_equal(x_new, data)
            np.testing.assert_equal(rowsize_new, rowsize)

    def test_prune_all_smaller(self):
        x = [1, 2, 3, 1, 2, 1, 2, 3, 4]
        rowsize = [3, 2, 4]
        minimum = 5

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, rowsize_new = prune(data, rowsize, minimum)
            np.testing.assert_equal(x_new, np.array([]))
            np.testing.assert_equal(rowsize_new, np.array([]))

    def test_prune_dates(self):
        a = pd.date_range(
            start=pd.to_datetime("1/1/2018"),
            end=pd.to_datetime("1/03/2018"),
        )

        b = pd.date_range(
            start=pd.to_datetime("1/1/2018"),
            end=pd.to_datetime("1/05/2018"),
        )

        c = pd.date_range(
            start=pd.to_datetime("1/1/2018"),
            end=pd.to_datetime("1/08/2018"),
        )

        x = np.concatenate((a, b, c))
        rowsize = [len(v) for v in [a, b, c]]

        x_new, rowsize_new = prune(x, rowsize, 5)
        np.testing.assert_equal(x_new, np.concatenate((b, c)))
        np.testing.assert_equal(rowsize_new, [5, 8])

    def test_prune_keep_nan(self):
        x = [1, 2, np.nan, 1, 2, 1, 2, np.nan, 4]
        rowsize = [3, 2, 4]
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, rowsize_new = prune(data, rowsize, minimum)
            np.testing.assert_equal(x_new, [1, 2, np.nan, 1, 2, np.nan, 4])
            np.testing.assert_equal(rowsize_new, [3, 4])

    def test_prune_empty(self):
        x = []
        rowsize = []
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            with self.assertRaises(IndexError):
                x_new, rowsize_new = prune(data, rowsize, minimum)

    def test_print_incompatible_rowsize(self):
        x = [1, 2, 3, 1, 2]
        rowsize = [3, 3]
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            with self.assertRaises(ValueError):
                x_new, rowsize_new = prune(data, rowsize, minimum)


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


class ragged_to_regular_tests(unittest.TestCase):
    def test_ragged_to_regular(self):
        ragged = np.array([1, 2, 3, 4, 5])
        rowsize = [2, 1, 2]
        expected = np.array([[1, 2], [3, np.nan], [4, 5]])

        result = ragged_to_regular(ragged, rowsize)
        self.assertTrue(np.all(np.isnan(result) == np.isnan(expected)))
        self.assertTrue(
            np.all(result[~np.isnan(result)] == expected[~np.isnan(expected)])
        )

        result = ragged_to_regular(
            xr.DataArray(data=ragged), xr.DataArray(data=rowsize)
        )
        self.assertTrue(np.all(np.isnan(result) == np.isnan(expected)))
        self.assertTrue(
            np.all(result[~np.isnan(result)] == expected[~np.isnan(expected)])
        )

        result = ragged_to_regular(pd.Series(data=ragged), pd.Series(data=rowsize))
        self.assertTrue(np.all(np.isnan(result) == np.isnan(expected)))
        self.assertTrue(
            np.all(result[~np.isnan(result)] == expected[~np.isnan(expected)])
        )

    def test_ragged_to_regular_fill_value(self):
        ragged = np.array([1, 2, 3, 4, 5])
        rowsize = [2, 1, 2]
        expected = np.array([[1, 2], [3, -999], [4, 5]])

        result = ragged_to_regular(ragged, rowsize, fill_value=-999)
        self.assertTrue(np.all(result == expected))

    def test_regular_to_ragged(self):
        regular = np.array([[1, 2], [3, np.nan], [4, 5]])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_rowsize = np.array([2, 1, 2])

        result, rowsize = regular_to_ragged(regular)
        self.assertTrue(np.all(result == expected))
        self.assertTrue(np.all(rowsize == expected_rowsize))

    def test_regular_to_ragged_fill_value(self):
        regular = np.array([[1, 2], [3, -999], [4, 5]])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_rowsize = np.array([2, 1, 2])

        result, rowsize = regular_to_ragged(regular, fill_value=-999)
        self.assertTrue(np.all(result == expected))
        self.assertTrue(np.all(rowsize == expected_rowsize))

    def test_ragged_to_regular_roundtrip(self):
        ragged = np.array([1, 2, 3, 4, 5])
        rowsize = [2, 1, 2]
        new_ragged, new_rowsize = regular_to_ragged(ragged_to_regular(ragged, rowsize))
        self.assertTrue(np.all(new_ragged == ragged))
        self.assertTrue(np.all(new_rowsize == rowsize))


class apply_ragged_tests(unittest.TestCase):
    def setUp(self):
        self.rowsize = [2, 3, 4]
        self.x = np.array([1, 2, 10, 12, 14, 30, 33, 36, 39])
        self.y = np.arange(0, len(self.x))
        self.t = np.array([1, 2, 1, 2, 3, 1, 2, 3, 4])

    def test_simple(self):
        y = apply_ragged(
            lambda x: x**2,
            np.array([1, 2, 3, 4]),
            [2, 2],
        )
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_simple_dataarray(self):
        y = apply_ragged(
            lambda x: x**2,
            xr.DataArray(data=[1, 2, 3, 4], coords={"obs": [1, 2, 3, 4]}),
            [2, 2],
        )
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_simple_with_args(self):
        y = apply_ragged(
            lambda x, p: x**p,
            np.array([1, 2, 3, 4]),
            [2, 2],
            2,
        )
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_simple_with_kwargs(self):
        y = apply_ragged(
            lambda x, p: x**p,
            np.array([1, 2, 3, 4]),
            [2, 2],
            p=2,
        )
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_with_rows(self):
        y = apply_ragged(
            lambda x: x**2,
            np.array([1, 2, 3, 4]),
            [2, 2],
            rows=0,
        )
        self.assertTrue(np.all(y == np.array([1, 4])))

        y = apply_ragged(
            lambda x: x**2,
            np.array([1, 2, 3, 4]),
            [2, 2],
            rows=[0, 1],
        )
        self.assertTrue(np.all(y == np.array([1, 4, 9, 16])))

    def test_with_axis(self):
        # Test that axis=0 is the default.
        x = np.arange((6)).reshape((3, 2))
        func = lambda x: x**2
        rowsize = [2, 1]
        y = apply_ragged(func, x, rowsize, axis=0)
        self.assertTrue(np.all(y == apply_ragged(func, x, rowsize)))

        # Test that the rowsize is checked against the correct axis.
        with self.assertRaises(ValueError):
            y = apply_ragged(func, x.T, rowsize, axis=0)

        # Test that applying an element-wise function on a 2-d array over
        # ragged axis 1 is th same as applying it to the transpose over ragged
        # axis 0.
        rowsize = [1, 1]
        y0 = apply_ragged(func, x.T, rowsize, axis=0)
        y1 = apply_ragged(func, x, rowsize, axis=1)
        self.assertTrue(np.all(y0 == y1.T))

        # Test that axis=1 works with reduction over the non-ragged axis.
        y = apply_ragged(np.sum, x, rowsize, axis=1)
        self.assertTrue(np.all(y == np.sum(x, axis=0)))

        # Test that the same works with xr.DataArray as input
        # (this did not work before the axis feature was added).
        y = apply_ragged(np.sum, xr.DataArray(data=x), rowsize, axis=1)
        self.assertTrue(np.all(y == np.sum(x, axis=0)))

        # Test that axis works for multiple outputs:
        func = lambda x: (np.mean(x), np.std(x))
        y = apply_ragged(func, x, rowsize, axis=1)
        self.assertTrue(np.all(y[0] == np.mean(x, axis=0)))
        self.assertTrue(np.all(y[1] == np.std(x, axis=0)))

    def test_velocity_ndarray(self):
        for executor in [futures.ThreadPoolExecutor(), futures.ProcessPoolExecutor()]:
            u, v = apply_ragged(
                velocity_from_position,
                [self.x, self.y, self.t],
                self.rowsize,
                coord_system="cartesian",
                executor=executor,
            )
            self.assertIsNone(
                np.testing.assert_allclose(
                    u, [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
                )
            )
            self.assertIsNone(
                np.testing.assert_allclose(
                    v, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                )
            )

    def test_velocity_dataarray(self):
        for executor in [futures.ThreadPoolExecutor(), futures.ProcessPoolExecutor()]:
            u, v = apply_ragged(
                velocity_from_position,
                [
                    xr.DataArray(data=self.x),
                    xr.DataArray(data=self.y),
                    xr.DataArray(data=self.t),
                ],
                xr.DataArray(data=self.rowsize),
                coord_system="cartesian",
                executor=executor,
            )
            self.assertIsNone(
                np.testing.assert_allclose(
                    u, [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
                )
            )
            self.assertIsNone(
                np.testing.assert_allclose(
                    v, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                )
            )

    def test_bad_rowsize_raises(self):
        with self.assertRaises(ValueError):
            for use_threads in [True, False]:
                apply_ragged(
                    lambda x: x**2,
                    np.array([1, 2, 3, 4]),
                    [2],
                    use_threads=use_threads,
                )


class subset_tests(unittest.TestCase):
    def setUp(self):
        self.ds = sample_ragged_array().to_xarray()

    def test_ds_unmodified(self):
        ds_original = self.ds.copy(deep=True)
        subset(self.ds, {"test": True})
        xr.testing.assert_equal(ds_original, self.ds)

    def test_equal(self):
        ds_sub = subset(self.ds, {"test": True})
        self.assertEqual(len(ds_sub.id), 2)

    def test_select(self):
        ds_sub = subset(self.ds, {"id": [1, 2]})
        self.assertTrue(all(ds_sub.id == [1, 2]))
        self.assertEqual(len(ds_sub.id), 2)

    def test_range(self):
        # positive
        ds_sub = subset(self.ds, {"lon": (0, 180)})
        traj_idx = np.insert(np.cumsum(ds_sub["rowsize"].values), 0, 0)
        self.assertTrue(
            all(ds_sub.lon[slice(traj_idx[0], traj_idx[1])] == [51, 61, 71])
        )

        self.assertTrue(all(ds_sub.lon[slice(traj_idx[1], traj_idx[2])] == [103, 113]))

        self.assertTrue(
            all(ds_sub.lon[slice(traj_idx[2], traj_idx[3])] == [12, 22, 32, 42])
        )

        # negative range
        ds_sub = subset(self.ds, {"lon": (-180, 0)})
        traj_idx = np.insert(np.cumsum(ds_sub["rowsize"].values), 0, 0)
        self.assertEqual(len(ds_sub.id), 1)
        self.assertEqual(ds_sub.id[0], 1)
        self.assertTrue(all(ds_sub.lon == [-121, -111]))

        # both
        ds_sub = subset(self.ds, {"lon": (-30, 30)})
        traj_idx = np.insert(np.cumsum(ds_sub["rowsize"].values), 0, 0)
        self.assertEqual(len(ds_sub.id), 1)
        self.assertEqual(ds_sub.id[0], 2)
        self.assertTrue(all(ds_sub.lon[slice(traj_idx[0], traj_idx[1])] == ([12, 22])))

    def test_combine(self):
        ds_sub = subset(
            self.ds, {"id": [1, 2], "lat": (-90, 20), "lon": (-180, 25), "test": True}
        )
        self.assertTrue(all(ds_sub.id == [1, 2]))
        self.assertTrue(all(ds_sub.lon == [-121, -111, 12]))
        self.assertTrue(all(ds_sub.lat == [-90, -45, 10]))

    def test_empty(self):
        with self.assertWarns(UserWarning):
            ds_sub = subset(self.ds, {"id": 3, "lon": (-180, 0)})
            self.assertTrue(len(ds_sub.sizes) == 0)

    def test_unknown_var(self):
        with self.assertRaises(ValueError):
            subset(self.ds, {"a": 10})

        with self.assertRaises(ValueError):
            subset(self.ds, {"lon": (0, 180), "a": (0, 10)})

    def test_ragged_array_with_id_as_str(self):
        ds_str = self.ds.copy()
        ds_str["id"].values = ds_str["id"].astype(str)

        ds_sub = subset(ds_str, {"id": ds_str["id"].values[0]})
        self.assertTrue(ds_sub["id"].size == 1)

        ds_sub = subset(ds_str, {"id": list(ds_str["id"].values[:2])})
        self.assertTrue(ds_sub["id"].size == 2)

    def test_ragged_array_with_id_as_object(self):
        ds_str = self.ds.copy()
        ds_str["id"].values = ds_str["id"].astype(object)

        ds_sub = subset(ds_str, {"id": ds_str["id"].values[0]})
        self.assertTrue(ds_sub["id"].size == 1)

        ds_sub = subset(ds_str, {"id": list(ds_str["id"].values[:2])})
        self.assertTrue(ds_sub["id"].size == 2)

    def test_arraylike_criterion(self):
        # DataArray
        ds_sub = subset(self.ds, {"id": self.ds["id"][:2]})
        self.assertTrue(ds_sub["id"].size == 2)

        # NumPy array
        ds_sub = subset(self.ds, {"id": self.ds["id"][:2].values})
        self.assertTrue(ds_sub["id"].size == 2)

    def test_full_rows(self):
        ds_id_rowsize = {
            i: j for i, j in zip(self.ds.id.values, self.ds.rowsize.values)
        }

        ds_sub = subset(self.ds, {"lon": (-125, -111)}, full_rows=True)
        self.assertTrue(all(ds_sub.lon == [-121, -111, 51, 61, 71]))

        ds_sub_id_rowsize = {
            i: j for i, j in zip(ds_sub.id.values, ds_sub.rowsize.values)
        }
        for k, v in ds_sub_id_rowsize.items():
            self.assertTrue(ds_id_rowsize[k] == v)

        ds_sub = subset(self.ds, {"lat": (30, 40)}, full_rows=True)
        self.assertTrue(all(ds_sub.lat == [10, 20, 30, 40]))

        ds_sub_id_rowsize = {
            i: j for i, j in zip(ds_sub.id.values, ds_sub.rowsize.values)
        }
        for k, v in ds_sub_id_rowsize.items():
            self.assertTrue(ds_id_rowsize[k] == v)

        ds_sub = subset(self.ds, {"time": (4, 5)}, full_rows=True)
        xr.testing.assert_equal(self.ds, ds_sub)

    def test_subset_by_rows(self):
        rows = [0, 2]  # test extracting first and third rows
        ds_sub = subset(self.ds, {"rows": rows})
        self.assertTrue(all(ds_sub["id"] == [1, 2]))
        self.assertTrue(all(ds_sub["rowsize"] == [5, 4]))

    def test_subset_callable(self):
        func = (
            lambda arr: ((arr - arr[0]) % 2) == 0
        )  # test keeping obs every two time intervals
        ds_sub = subset(self.ds, {"time": func})
        self.assertTrue(all(ds_sub["id"] == [1, 3, 2]))
        self.assertTrue(all(ds_sub["rowsize"] == [3, 1, 2]))
        self.assertTrue(all(ds_sub["time"] == [1, 3, 5, 4, 2, 4]))

        func = lambda arr: arr <= 2  # keep id larger or equal to 2
        ds_sub = subset(self.ds, {"id": func})
        self.assertTrue(all(ds_sub["id"] == [1, 2]))
        self.assertTrue(all(ds_sub["rowsize"] == [5, 4]))

    def test_subset_callable_tuple(self):
        func = lambda arr1, arr2: np.logical_and(
            arr1 >= 0, arr2 >= 30
        )  # keep positive longitude and latitude larger or equal than 30
        ds_sub = subset(self.ds, {("lon", "lat"): func})
        self.assertTrue(all(ds_sub["id"] == [1, 2]))
        self.assertTrue(all(ds_sub["rowsize"] == [2, 2]))
        self.assertTrue(all(ds_sub["lon"] >= 0))
        self.assertTrue(all(ds_sub["lat"] >= 30))

    def test_subset_callable_wrong_dim(self):
        func = lambda arr: [arr, arr]  # returns 2 values per element
        with self.assertRaises(ValueError):
            subset(self.ds, {"time": func})
        with self.assertRaises(ValueError):
            subset(self.ds, {"id": func})

    def test_subset_callable_wrong_type(self):
        rows = [0, 2]  # test extracting first and third rows
        with self.assertRaises(TypeError):  # passing a tuple when a string is expected
            subset(self.ds, {("rows",): rows})

    def test_subset_callable_tuple_unknown_var(self):
        func = lambda arr1, arr2: np.logical_and(
            arr1 >= 0, arr2 >= 30
        )  # keep positive longitude and latitude larger or equal than 30
        with self.assertRaises(ValueError):
            subset(self.ds, {("a", "lat"): func})

    def test_subset_callable_tuple_not_same_dimension(self):
        func = lambda arr1, arr2: np.logical_and(
            arr1 >= 0, arr2 >= 30
        )  # keep positive longitude and latitude larger or equal than 30
        with self.assertRaises(TypeError):
            subset(self.ds, {("id", "lat"): func})


class unpack_tests(unittest.TestCase):
    def test_unpack(self):
        ds = sample_ragged_array().to_xarray()

        # Test unpacking into DataArrays
        lon = unpack(ds.lon, ds["rowsize"])

        self.assertTrue(isinstance(lon, list))
        self.assertTrue(np.all([type(a) is xr.DataArray for a in lon]))
        self.assertTrue(
            np.all([lon[n].size == ds["rowsize"][n] for n in range(len(lon))])
        )

        # Test unpacking into np.ndarrays
        lon = unpack(ds.lon.values, ds["rowsize"])

        self.assertTrue(isinstance(lon, list))
        self.assertTrue(np.all([type(a) is np.ndarray for a in lon]))
        self.assertTrue(
            np.all([lon[n].size == ds["rowsize"][n] for n in range(len(lon))])
        )

    def test_unpack_rows(self):
        ds = sample_ragged_array().to_xarray()
        x = ds.lon.values
        rowsize = ds.rowsize.values

        self.assertTrue(
            all(
                np.array_equal(a, b)
                for a, b in zip(unpack(x, rowsize, None), unpack(x, rowsize))
            )
        )
        self.assertTrue(
            all(
                np.array_equal(a, b)
                for a, b in zip(unpack(x, rowsize, 0), unpack(x, rowsize)[:1])
            )
        )
        self.assertTrue(
            all(
                np.array_equal(a, b)
                for a, b in zip(unpack(x, rowsize, [0, 1]), unpack(x, rowsize)[:2])
            )
        )
        # Test that unpack can accept rows as numpy integer as well, not just
        # the built-in int.
        self.assertTrue(
            all(
                np.array_equal(a, b)
                for a, b in zip(unpack(x, rowsize, np.int64(0)), unpack(x, rowsize)[:1])
            )
        )

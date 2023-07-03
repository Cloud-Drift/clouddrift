from clouddrift.analysis import (
    apply_ragged,
    chunk,
    prune,
    position_from_velocity,
    ragged_to_regular,
    regular_to_ragged,
    segment,
    subset,
    unpack_ragged,
    velocity_from_position,
)
from clouddrift.sphere import EARTH_RADIUS_METERS
from clouddrift.raggedarray import RaggedArray
import unittest
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from concurrent import futures


if __name__ == "__main__":
    unittest.main()


def sample_ragged_array() -> RaggedArray:
    drifter_id = [1, 2, 3]
    count = [5, 4, 2]
    longitude = [[-121, -111, 51, 61, 71], [12, 22, 32, 42], [103, 113]]
    latitude = [[-90, -45, 45, 90, 0], [10, 20, 30, 40], [10, 20]]
    t = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [4, 5]]
    ids = [[1, 1, 1, 1, 1], [2, 2, 2, 2], [3, 3]]
    test = [
        [True, True, True, False, False],
        [True, False, False, False],
        [False, False],
    ]
    nb_obs = np.sum(count)
    nb_traj = len(drifter_id)
    attrs_global = {
        "title": "test trajectories",
        "history": "version xyz",
    }
    variables_coords = ["ids", "time", "lon", "lat"]

    coords = {"lon": longitude, "lat": latitude, "ids": ids, "time": t}
    metadata = {"ID": drifter_id, "count": count}
    data = {"test": test}

    # append xr.Dataset to a list
    list_ds = []
    for i in range(0, len(count)):
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

    ra = RaggedArray.from_files(
        [0, 1, 2],
        lambda i: list_ds[i],
        variables_coords,
        ["ID", "count"],
        ["test"],
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

    def test_chunk_count(self):
        # Simple chunk with count
        self.assertTrue(
            np.all(
                apply_ragged(chunk, np.array([1, 2, 3, 4, 5, 6]), [2, 3, 1], 2)
                == np.array([[1, 2], [3, 4]])
            )
        )

        # count with overlap
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
        count = [3, 2, 4]
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, count_new = prune(data, count, minimum)
            self.assertTrue(type(x_new) is np.ndarray)
            self.assertTrue(type(count_new) is np.ndarray)
            np.testing.assert_equal(x_new, [1, 2, 3, 1, 2, 3, 4])
            np.testing.assert_equal(count_new, [3, 4])

    def test_prune_all_longer(self):
        x = [1, 2, 3, 1, 2, 1, 2, 3, 4]
        count = [3, 2, 4]
        minimum = 1

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, count_new = prune(data, count, minimum)
            np.testing.assert_equal(x_new, data)
            np.testing.assert_equal(count_new, count)

    def test_prune_all_smaller(self):
        x = [1, 2, 3, 1, 2, 1, 2, 3, 4]
        count = [3, 2, 4]
        minimum = 5

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, count_new = prune(data, count, minimum)
            np.testing.assert_equal(x_new, np.array([]))
            np.testing.assert_equal(count_new, np.array([]))

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
        count = [len(v) for v in [a, b, c]]

        x_new, count_new = prune(x, count, 5)
        np.testing.assert_equal(x_new, np.concatenate((b, c)))
        np.testing.assert_equal(count_new, [5, 8])

    def test_prune_keep_nan(self):
        x = [1, 2, np.nan, 1, 2, 1, 2, np.nan, 4]
        count = [3, 2, 4]
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            x_new, count_new = prune(data, count, minimum)
            np.testing.assert_equal(x_new, [1, 2, np.nan, 1, 2, np.nan, 4])
            np.testing.assert_equal(count_new, [3, 4])

    def test_prune_empty(self):
        x = []
        count = []
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            with self.assertRaises(IndexError):
                x_new, count_new = prune(data, count, minimum)

    def test_print_incompatible_count(self):
        x = [1, 2, 3, 1, 2]
        count = [3, 3]
        minimum = 3

        for data in [x, np.array(x), pd.Series(data=x), xr.DataArray(data=x)]:
            with self.assertRaises(ValueError):
                x_new, count_new = prune(data, count, minimum)


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

    def test_segment_count(self):
        x = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
        tol = 0.5
        count = [6, 5]
        segment_sizes = segment(x, tol, count)
        self.assertTrue(type(segment_sizes) is np.ndarray)
        self.assertTrue(np.all(segment_sizes == np.array([1, 3, 2, 4, 1])))

    def test_segment_positive_and_negative_tolerance(self):
        x = [1, 1, 2, 2, 1, 1, 2, 2]
        segment_sizes = segment(x, 0.5, count=segment(x, -0.5))
        self.assertTrue(np.all(segment_sizes == np.array([2, 2, 2, 2])))

    def test_segment_count_raises(self):
        x = [0, 1, 2, 3]
        tol = 0.5
        count = [1, 2]  # count is too short
        with self.assertRaises(ValueError):
            segment(x, tol, count)

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
        count = [2, 1, 2]
        expected = np.array([[1, 2], [3, np.nan], [4, 5]])

        result = ragged_to_regular(ragged, count)
        self.assertTrue(np.all(np.isnan(result) == np.isnan(expected)))
        self.assertTrue(
            np.all(result[~np.isnan(result)] == expected[~np.isnan(expected)])
        )

        result = ragged_to_regular(xr.DataArray(data=ragged), xr.DataArray(data=count))
        self.assertTrue(np.all(np.isnan(result) == np.isnan(expected)))
        self.assertTrue(
            np.all(result[~np.isnan(result)] == expected[~np.isnan(expected)])
        )

        result = ragged_to_regular(pd.Series(data=ragged), pd.Series(data=count))
        self.assertTrue(np.all(np.isnan(result) == np.isnan(expected)))
        self.assertTrue(
            np.all(result[~np.isnan(result)] == expected[~np.isnan(expected)])
        )

    def test_ragged_to_regular_fill_value(self):
        ragged = np.array([1, 2, 3, 4, 5])
        count = [2, 1, 2]
        expected = np.array([[1, 2], [3, -999], [4, 5]])

        result = ragged_to_regular(ragged, count, fill_value=-999)
        self.assertTrue(np.all(result == expected))

    def test_regular_to_ragged(self):
        regular = np.array([[1, 2], [3, np.nan], [4, 5]])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_count = np.array([2, 1, 2])

        result, count = regular_to_ragged(regular)
        self.assertTrue(np.all(result == expected))
        self.assertTrue(np.all(count == expected_count))

    def test_regular_to_ragged_fill_value(self):
        regular = np.array([[1, 2], [3, -999], [4, 5]])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_count = np.array([2, 1, 2])

        result, count = regular_to_ragged(regular, fill_value=-999)
        self.assertTrue(np.all(result == expected))
        self.assertTrue(np.all(count == expected_count))

    def test_ragged_to_regular_roundtrip(self):
        ragged = np.array([1, 2, 3, 4, 5])
        count = [2, 1, 2]
        new_ragged, new_count = regular_to_ragged(ragged_to_regular(ragged, count))
        self.assertTrue(np.all(new_ragged == ragged))
        self.assertTrue(np.all(new_count == count))


class position_from_velocity_tests(unittest.TestCase):
    def setUp(self):
        self.INPUT_SIZE = 100
        self.lon = np.rad2deg(
            np.linspace(-np.pi, np.pi, self.INPUT_SIZE, endpoint=False)
        )
        self.lat = np.linspace(0, 45, self.INPUT_SIZE)
        self.time = np.linspace(0, 1e7, self.INPUT_SIZE)
        self.uf, self.vf = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="forward"
        )
        self.ub, self.vb = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="backward"
        )
        self.uc, self.vc = velocity_from_position(
            self.lon, self.lat, self.time, difference_scheme="centered"
        )

    def test_result_has_same_size_as_input(self):
        lon, lat = position_from_velocity(
            self.uf,
            self.vf,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.all(self.uf.shape == lon.shape))
        self.assertTrue(np.all(self.uf.shape == lat.shape))

    def test_velocity_position_roundtrip_forward(self):
        lon, lat = position_from_velocity(
            self.uf,
            self.vf,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.allclose(lon, self.lon))
        self.assertTrue(np.allclose(lat, self.lat))

    def test_velocity_position_roundtrip_backward(self):
        lon, lat = position_from_velocity(
            self.ub,
            self.vb,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="backward",
        )
        self.assertTrue(np.allclose(lon, self.lon))
        self.assertTrue(np.allclose(lat, self.lat))

    def test_velocity_position_roundtrip_centered(self):
        lon, lat = position_from_velocity(
            self.uc,
            self.vc,
            self.time,
            self.lon[0],
            self.lat[0],
            integration_scheme="centered",
        )
        # Centered scheme damps the 2dx waves so we need a looser tolerance.
        self.assertTrue(np.allclose(lon, self.lon, atol=1e-2))
        self.assertTrue(np.allclose(lat, self.lat, atol=1e-2))

    def test_time_axis(self):
        uf = np.transpose(
            np.reshape(np.tile(self.uf, 4), (2, 2, self.uf.size)), (0, 2, 1)
        )
        vf = np.transpose(
            np.reshape(np.tile(self.vf, 4), (2, 2, self.vf.size)), (0, 2, 1)
        )
        time = np.transpose(
            np.reshape(np.tile(self.time, 4), (2, 2, self.time.size)), (0, 2, 1)
        )
        expected_lon = np.transpose(
            np.reshape(np.tile(self.lon, 4), (2, 2, self.lon.size)), (0, 2, 1)
        )
        expected_lat = np.transpose(
            np.reshape(np.tile(self.lat, 4), (2, 2, self.lat.size)), (0, 2, 1)
        )
        lon, lat = position_from_velocity(
            uf,
            vf,
            time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
            time_axis=1,
        )
        self.assertTrue(np.allclose(lon, expected_lon))
        self.assertTrue(np.allclose(lat, expected_lat))
        self.assertTrue(np.all(lon.shape == expected_lon.shape))
        self.assertTrue(np.all(lat.shape == expected_lat.shape))

    def test_works_with_xarray(self):
        lon, lat = position_from_velocity(
            xr.DataArray(data=self.uf),
            xr.DataArray(data=self.vf),
            xr.DataArray(data=self.time),
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.allclose(lon, self.lon))
        self.assertTrue(np.allclose(lat, self.lat))

    def test_works_with_2d_array(self):
        uf = np.reshape(np.tile(self.uf, 4), (4, self.uf.size))
        vf = np.reshape(np.tile(self.vf, 4), (4, self.vf.size))
        time = np.reshape(np.tile(self.time, 4), (4, self.time.size))
        expected_lon = np.reshape(np.tile(self.lon, 4), (4, self.lon.size))
        expected_lat = np.reshape(np.tile(self.lat, 4), (4, self.lat.size))
        lon, lat = position_from_velocity(
            uf,
            vf,
            time,
            self.lon[0],
            self.lat[0],
            integration_scheme="forward",
        )
        self.assertTrue(np.allclose(lon, expected_lon))
        self.assertTrue(np.allclose(lat, expected_lat))
        self.assertTrue(np.allclose(lon.shape, expected_lon.shape))
        self.assertTrue(np.allclose(lon.shape, expected_lat.shape))


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

    def test_works_with_xarray(self):
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
        self.count = [2, 3, 4]
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

    def test_velocity_ndarray(self):
        for executor in [futures.ThreadPoolExecutor(), futures.ProcessPoolExecutor()]:
            u, v = apply_ragged(
                velocity_from_position,
                [self.x, self.y, self.t],
                self.count,
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
                xr.DataArray(data=self.count),
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

    def test_bad_count_raises(self):
        with self.assertRaises(ValueError):
            for use_threads in [True, False]:
                y = apply_ragged(
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
        ds_sub = subset(self.ds, {"test": True})
        xr.testing.assert_equal(ds_original, self.ds)

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
        traj_idx = np.insert(np.cumsum(ds_sub["count"].values), 0, 0)
        self.assertTrue(
            all(ds_sub.lon[slice(traj_idx[0], traj_idx[1])] == [51, 61, 71])
        )
        self.assertTrue(
            all(ds_sub.lon[slice(traj_idx[1], traj_idx[2])] == [12, 22, 32, 42])
        )
        self.assertTrue(all(ds_sub.lon[slice(traj_idx[2], traj_idx[3])] == [103, 113]))

        # negative range
        ds_sub = subset(self.ds, {"lon": (-180, 0)})
        traj_idx = np.insert(np.cumsum(ds_sub["count"].values), 0, 0)
        self.assertEqual(len(ds_sub.ID), 1)
        self.assertEqual(ds_sub.ID[0], 1)
        self.assertTrue(all(ds_sub.lon == [-121, -111]))

        # both
        ds_sub = subset(self.ds, {"lon": (-30, 30)})
        traj_idx = np.insert(np.cumsum(ds_sub["count"].values), 0, 0)
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


class unpack_ragged_tests(unittest.TestCase):
    def test_unpack_ragged(self):
        ds = sample_ragged_array().to_xarray()

        # Test unpacking into DataArrays
        lon = unpack_ragged(ds.lon, ds["count"])

        self.assertTrue(type(lon) is list)
        self.assertTrue(np.all([type(a) is xr.DataArray for a in lon]))
        self.assertTrue(
            np.all([lon[n].size == ds["count"][n] for n in range(len(lon))])
        )

        # Test unpacking into np.ndarrays
        lon = unpack_ragged(ds.lon.values, ds["count"])

        self.assertTrue(type(lon) is list)
        self.assertTrue(np.all([type(a) is np.ndarray for a in lon]))
        self.assertTrue(
            np.all([lon[n].size == ds["count"][n] for n in range(len(lon))])
        )

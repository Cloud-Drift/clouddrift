import datetime
import functools
import unittest

import numpy as np

from clouddrift.binning import (
    DEFAULT_BINS_NUMBER,
    DEFAULT_COORD_NAME,
    DEFAULT_DATA_NAME,
    _is_datetime_array,
    binned_statistics,
)


class binning_tests(unittest.TestCase):
    def setUp(self):
        self.coords_1d = np.array(
            [-0.2, 0.3, 0.6, 0.7, 1.2, 1.3, 1.8, 2.1, 2.7, 2.8, 3.1]
        )
        self.date_1d = np.datetime64("2020-01-01 00:00") + np.arange(
            11
        ) * np.timedelta64(1, "D")

        self.values_1d = np.array([1, 2, 3, 4, 5, 6, 9, 20, 20, 20, 20])
        self.bins_range_1d = (0, 3)

        self.coords_2d = [self.coords_1d, self.coords_1d]
        self.bins_range_2d = [(0, 3), (0, 3)]

        self.coords_3d = [self.coords_1d, self.coords_1d, self.coords_1d]
        self.bins_range_3d = [(0, 3), (0, 3), (0, 3)]

        # example with 8 regions around x=[-1,0,1], y=[-1,0,1], z=[-1,0,1]
        self.coords_3d_ex = [[], [], []]
        self.values_3d_ex = []
        for x in [-0.5, -0.25, 0.25, 0.5]:
            for y in [-0.5, -0.25, 0.25, 0.5]:
                for z in [-0.5, -0.25, 0.25, 0.5]:
                    self.coords_3d_ex[0].append(x)
                    self.coords_3d_ex[1].append(y)
                    self.coords_3d_ex[2].append(z)
                    self.values_3d_ex.append(int(x > 0) + int(y > 0) + int(z > 0))

    def test_parameters_dimensions(self):
        with self.assertRaises(ValueError):
            binned_statistics(self.coords_1d, dim_names=["x", "y"])

        with self.assertRaises(ValueError):
            binned_statistics(self.coords_1d, bins=[10, 10])

        with self.assertRaises(ValueError):
            binned_statistics(self.coords_2d, bins=[10])

        with self.assertRaises(ValueError):
            binned_statistics(self.coords_2d, dim_names=["x"])

        with self.assertRaises(ValueError):
            binned_statistics(self.coords_1d, output_names=["x", "y"])

        with self.assertRaises(ValueError):
            binned_statistics(
                self.coords_1d,
                data=np.ones_like(self.coords_1d),
                output_names=["x", "y"],
            )
        with self.assertRaises(ValueError):
            binned_statistics(self.coords_1d, data=np.ones((len(self.coords_1d), 1)))

    def test_is_datetime(self):
        arr = np.array([np.datetime64("2020-01-01"), np.datetime64("2020-01-02")])
        self.assertTrue(_is_datetime_array(arr))

        arr = np.array(
            [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")], dtype="O"
        )
        self.assertTrue(_is_datetime_array(arr))

        arr = np.array([datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)])
        self.assertTrue(_is_datetime_array(arr))

        arr = np.array([1, 2, 3])
        self.assertFalse(_is_datetime_array(arr))

        arr = np.array([1, 2, 3], dtype="O")
        self.assertFalse(_is_datetime_array(arr))

        arr = np.array([None], dtype="O")
        self.assertFalse(_is_datetime_array(arr))

        arr = np.array([np.nan, np.nan])
        self.assertFalse(_is_datetime_array(arr))

    def test_bins_number_default(self):
        ds = binned_statistics(self.coords_1d)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), DEFAULT_BINS_NUMBER)

        ds = binned_statistics(self.coords_2d)
        for v in ds.sizes.values():
            self.assertEqual(v, DEFAULT_BINS_NUMBER)

        ds = binned_statistics(self.coords_3d, bins=5)
        for v in ds.sizes.values():
            self.assertEqual(v, 5)

        ds = binned_statistics(self.coords_2d, bins=(5, None))
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), 5)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_1"]), DEFAULT_BINS_NUMBER)

    def test_bins_list(self):
        ds = binned_statistics(self.coords_1d, bins=[[0, 1, 2, 3]])
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), 3)

        ds = binned_statistics(self.coords_1d, bins=[np.arange(0, 4, 1)])
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), 3)

        ds = binned_statistics(self.coords_1d, bins=[np.arange(0, 4, 0.5)])
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), 7)

    def test_1d_hist_number(self):
        ds = binned_statistics(self.coords_1d, bins=3)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), 3)

    def test_2d_hist_number(self):
        ds = binned_statistics(self.coords_2d, bins=(3, 4))
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), 3)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_1"]), 4)

    def test_3d_hist_number(self):
        ds = binned_statistics(self.coords_3d, bins=(3, 4, 5))
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), 3)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_1"]), 4)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_2"]), 5)

    def test_hist_center(self):
        for i in range(1, 10):
            ds = binned_statistics(self.coords_1d, bins=i)

            bins_coords = np.linspace(
                np.min(self.coords_1d), np.max(self.coords_1d), i + 1
            )
            bins_center = (bins_coords[:-1] + bins_coords[1:]) / 2

            self.assertIsNone(
                np.testing.assert_allclose(
                    ds[f"{DEFAULT_COORD_NAME}_0"].values, bins_center
                )
            )

    def test_1d_output(self):
        ds = binned_statistics(coords=self.coords_1d)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), DEFAULT_BINS_NUMBER)
        self.assertEqual(
            sum(ds[f"{DEFAULT_DATA_NAME}_count"].values), len(self.coords_1d)
        )

    def test_1d_output_bins(self):
        n_bins = 3
        ds = binned_statistics(coords=self.coords_1d, bins=n_bins)
        self.assertEqual(len(ds[f"{DEFAULT_COORD_NAME}_0"]), n_bins)
        self.assertEqual(
            sum(ds[f"{DEFAULT_DATA_NAME}_count"].values), len(self.coords_1d)
        )
        self.assertEqual(len(ds[f"{DEFAULT_DATA_NAME}_count"].shape), 1)

    def test_1d_output_variables_mean(self):
        value_1 = 1
        value_2 = 2
        variable = [
            np.ones_like(self.coords_1d) * value_1,
            np.ones_like(self.coords_1d) * value_2,
        ]
        ds = binned_statistics(
            coords=self.coords_1d, data=variable, bins=2, statistics="mean"
        )
        self.assertEqual(len(ds.data_vars), 2)
        self.assertTrue(all(ds[f"{DEFAULT_DATA_NAME}_0_mean"] == value_1))
        self.assertTrue(all(ds[f"{DEFAULT_DATA_NAME}_1_mean"] == value_2))

        coords_threshold = 1
        variables = np.ones_like(self.coords_1d) * value_1
        variables[self.coords_1d > coords_threshold] = value_2
        ds = binned_statistics(
            coords=self.coords_1d,
            data=variables,
            bins=3,
            bins_range=(0, 3),
            statistics="mean",
        )
        mask = ds[f"{DEFAULT_COORD_NAME}_0"].values < coords_threshold
        self.assertTrue(all(ds[f"{DEFAULT_DATA_NAME}_0_mean"].values[mask] == value_1))
        self.assertTrue(all(ds[f"{DEFAULT_DATA_NAME}_0_mean"].values[~mask] == value_2))

    def test_2d_output(self):
        ds = binned_statistics(coords=self.coords_2d)
        for v in ds.sizes.values():
            self.assertEqual(v, DEFAULT_BINS_NUMBER)
        self.assertEqual(len(ds[f"{DEFAULT_DATA_NAME}_count"].shape), 2)

        n_bins = (3, 4)
        ds = binned_statistics(coords=self.coords_2d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(v, n_bins[i])
        self.assertEqual(len(ds[f"{DEFAULT_DATA_NAME}_count"].shape), 2)

        n_bins = (3, None)
        ds = binned_statistics(coords=self.coords_2d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(
                v, n_bins[i] if n_bins[i] is not None else DEFAULT_BINS_NUMBER
            )
        self.assertEqual(len(ds[f"{DEFAULT_DATA_NAME}_count"].shape), 2)

    def test_3d_output(self):
        ds = binned_statistics(coords=self.coords_3d)
        for v in ds.sizes.values():
            self.assertEqual(v, DEFAULT_BINS_NUMBER)
        self.assertEqual(len(ds[f"{DEFAULT_DATA_NAME}_count"].shape), 3)

        n_bins = (3, 4, 5)
        ds = binned_statistics(coords=self.coords_3d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(v, n_bins[i])
        self.assertEqual(len(ds[f"{DEFAULT_DATA_NAME}_count"].shape), 3)

        n_bins = (3, 4, None)
        ds = binned_statistics(coords=self.coords_3d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(
                v, n_bins[i] if n_bins[i] is not None else DEFAULT_BINS_NUMBER
            )
        self.assertEqual(len(ds[f"{DEFAULT_DATA_NAME}_count"].shape), 3)

    def test_3d_output_mean_example(self):
        ds = binned_statistics(
            coords=self.coords_3d_ex,
            data=self.values_3d_ex,
            bins=(2, 2, 2),
            bins_range=[(-1, 1), (-1, 1), (-1, 1)],
            statistics="mean",
        )
        self.assertEqual(len(ds.data_vars), 1)
        self.assertIsNone(
            np.testing.assert_allclose(
                ds[f"{DEFAULT_DATA_NAME}_0_mean"].values.flatten(),
                np.array(
                    [
                        0,  # (-1, -1, -1)
                        1,  # (-1, -1, 0)
                        1,  # (-1, 0, -1)
                        2,  # (-1, 0, 0)
                        1,  # (0, -1, -1)
                        2,  # (0, -1, 0)
                        2,  # (0, 0, -1)
                        3,  # (0, 0, 0)
                    ]
                ),
            )
        )

    def test_hist_range(self):
        ds = binned_statistics(
            coords=self.coords_1d, bins=3, bins_range=self.bins_range_1d
        )
        self.assertEqual(
            sum(ds[f"{DEFAULT_DATA_NAME}_count"].values),
            len(
                self.coords_1d[
                    np.logical_and(
                        self.coords_1d >= self.bins_range_1d[0],
                        self.coords_1d < self.bins_range_1d[1],
                    )
                ]
            ),
        )

    def test_hist_range_2d(self):
        ds = binned_statistics(
            coords=self.coords_2d, bins=(3, 3), bins_range=self.bins_range_2d
        )
        self.assertEqual(
            sum(ds[f"{DEFAULT_DATA_NAME}_count"].values.flatten()),
            len(
                self.coords_1d[
                    np.logical_and(
                        self.coords_1d >= self.bins_range_1d[0],
                        self.coords_1d < self.bins_range_1d[1],
                    )
                ]
            ),
        )

    def test_hist_range_3d(self):
        ds = binned_statistics(
            coords=self.coords_3d, bins=(3, 3, 3), bins_range=self.bins_range_3d
        )
        self.assertEqual(
            sum(ds[f"{DEFAULT_DATA_NAME}_count"].values.flatten()),
            len(
                self.coords_1d[
                    np.logical_and(
                        self.coords_1d >= self.bins_range_1d[0],
                        self.coords_1d < self.bins_range_1d[1],
                    )
                ]
            ),
        )

    def test_rename_dimensions(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            bins=4,
            dim_names=["x"],
        )
        self.assertIn("x", ds.sizes)

        ds = binned_statistics(
            coords=self.coords_2d,
            bins=(3, 4),
            dim_names=["x", "y"],
        )
        self.assertIn("x", ds.sizes)
        self.assertIn("y", ds.sizes)

    def test_coords_finite(self):
        coords_inf = self.coords_1d.copy()
        coords_inf[0] = np.inf

        with self.assertRaises(ValueError):
            binned_statistics(
                coords=coords_inf,
                bins=3,
                data=self.values_1d,
            )

        coords_nan = [self.coords_1d.copy(), self.coords_1d.copy()]
        coords_nan[1][0] = np.nan
        with self.assertRaises(ValueError):
            binned_statistics(
                coords=coords_nan,
                bins=3,
                data=self.values_1d,
            )

        coords_nan = [self.date_1d.copy()]
        coords_nan[0][0] = np.datetime64("NaT")
        with self.assertRaises(ValueError):
            binned_statistics(
                coords=coords_nan,
                bins=3,
                data=self.values_1d,
            )

    def test_rename_variables(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            bins=4,
            output_names=["x"],
        )
        self.assertIn("x_count", ds.data_vars)

        ds = binned_statistics(
            coords=self.coords_2d,
            data=[self.coords_1d, self.coords_1d],
            output_names=["x", "y"],
        )
        self.assertIn("x_count", ds.data_vars)
        self.assertIn("y_count", ds.data_vars)

        ds = binned_statistics(
            coords=self.coords_2d,
            data=[self.coords_1d, self.coords_1d],
            output_names=["x", "y"],
            statistics=["count", "mean"],
        )
        self.assertIn("x_count", ds.data_vars)
        self.assertIn("y_count", ds.data_vars)
        self.assertIn("x_mean", ds.data_vars)
        self.assertIn("y_mean", ds.data_vars)

    def test_statistics_wrong_function(self):
        with self.assertRaises(ValueError):
            binned_statistics(
                coords=self.coords_1d,
                data=self.values_1d,
                bins=3,
                statistics=["non_existent_function"],
            )

    def test_statistics_wrong_type(self):
        with self.assertRaises(ValueError):
            binned_statistics(
                coords=self.coords_1d,
                data=self.values_1d,
                bins=3,
                statistics=["mean", 42],
            )

        with self.assertRaises(ValueError):
            binned_statistics(
                coords=self.coords_1d,
                data=self.values_1d,
                bins=3,
                statistics=np.array([1, 2, 3]),
            )

    def test_statistics_default(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_count", ds.data_vars)

    def test_statistics_no_precalculated_values(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
            statistics="mean",
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean", ds.data_vars)

        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
            statistics="std",
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_std", ds.data_vars)

    def test_statistics_all(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
            statistics=["count", "sum", "mean", "std", "min", "max"],
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_count", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_sum", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_std", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_min", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_max", ds.data_vars)

    def test_statistics_n_vars(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=[self.values_1d, self.values_1d, self.values_1d],
            statistics=["count", "sum", "mean", "std", "min", "max"],
        )
        for var in [f"{DEFAULT_DATA_NAME}_0", f"{DEFAULT_DATA_NAME}_1"]:
            self.assertIn(f"{var}_count", ds.data_vars)
            self.assertIn(f"{var}_sum", ds.data_vars)
            self.assertIn(f"{var}_mean", ds.data_vars)
            self.assertIn(f"{var}_std", ds.data_vars)
            self.assertIn(f"{var}_min", ds.data_vars)
            self.assertIn(f"{var}_max", ds.data_vars)

    def test_statistics_n_vars_rename(self):
        var_names = ["u", "v", "w"]
        ds = binned_statistics(
            coords=self.coords_1d,
            data=[self.values_1d, self.values_1d, self.values_1d],
            statistics=["count", "sum", "mean", "median", "std", "min", "max"],
            output_names=var_names,
        )
        for var in var_names:
            self.assertIn(f"{var}_count", ds.data_vars)
            self.assertIn(f"{var}_sum", ds.data_vars)
            self.assertIn(f"{var}_mean", ds.data_vars)
            self.assertIn(f"{var}_median", ds.data_vars)
            self.assertIn(f"{var}_std", ds.data_vars)
            self.assertIn(f"{var}_min", ds.data_vars)
            self.assertIn(f"{var}_max", ds.data_vars)

    def test_statistics_callable(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
            statistics=[np.median, "mean"],
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_median", ds.data_vars)

        np.testing.assert_allclose(
            ds[f"{DEFAULT_DATA_NAME}_0_mean"].values,
            np.array([2.5, 6.666, 20.0]),
            rtol=1e-2,
        )

        np.testing.assert_allclose(
            ds[f"{DEFAULT_DATA_NAME}_0_median"].values,
            np.array([2.5, 6, 20.0]),
            rtol=1e-2,
        )

    def test_statistics_callable_partial(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
            statistics=[
                functools.partial(np.percentile, q=25),
                functools.partial(np.percentile, q=50),
                functools.partial(np.percentile, q=75),
                "mean",
            ],
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_percentile", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_percentile_1", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_percentile_2", ds.data_vars)

    def test_statistics_callable_lambda(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
            statistics=[lambda x: np.percentile(x, q=40), "mean"],
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_stat", ds.data_vars)

    def test_statistics_callable_multiple_lambda(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=self.values_1d,
            bins=3,
            statistics=[
                lambda x: np.percentile(x, q=10),
                lambda x: np.percentile(x, q=20),
                lambda x: np.percentile(x, q=30),
                np.mean,
                np.mean,
            ],
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_stat", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_stat_1", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_stat_2", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean_1", ds.data_vars)

    def test_statistics_multivariable_wrong_parameter(self):
        with self.assertRaises(ValueError):
            binned_statistics(
                coords=self.coords_1d,
                data=[self.values_1d, self.values_1d],
                bins=3,
                statistics=(
                    0,
                    lambda data: np.sqrt(np.mean(data[0] ** 2 + data[1] ** 2)),
                ),
            )

        with self.assertRaises(ValueError):
            binned_statistics(
                coords=self.coords_1d,
                data=[self.values_1d, self.values_1d],
                bins=3,
                statistics=(
                    "ke",
                    0,
                ),
            )

    def test_statistics_multivariable(self):
        ds = binned_statistics(
            coords=self.coords_1d,
            data=[np.ones_like(self.values_1d) * 3, np.ones_like(self.values_1d) * 4],
            bins=3,
            statistics=[
                "mean",
                "count",
                (
                    "ke",
                    lambda data: np.sqrt(np.mean(data[0] ** 2 + data[1] ** 2)),
                ),
            ],
        )
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_count", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_1_count", ds.data_vars)
        self.assertIn(f"{DEFAULT_DATA_NAME}_0_mean", ds.data_vars)
        self.assertTrue(all(ds[f"{DEFAULT_DATA_NAME}_0_mean"] == 3))
        self.assertIn(f"{DEFAULT_DATA_NAME}_1_mean", ds.data_vars)
        self.assertTrue(all(ds[f"{DEFAULT_DATA_NAME}_1_mean"] == 4))
        self.assertIn("ke", ds.data_vars)
        self.assertTrue(all(ds["ke"] == 5))

    def test_statistics_complex_mean_sum_count_std(self):
        # Complex input for mean, sum, count, std
        coords = self.coords_1d
        values = np.array(
            [
                1 + 1j,
                2 + 2j,
                3 + 3j,
                4 + 4j,
                5 + 5j,
                6 + 6j,
                7 + 7j,
                8 + 8j,
                9 + 9j,
                10 + 10j,
                11 + 11j,
            ]
        )
        ds = binned_statistics(
            coords=coords,
            data=values,
            bins=3,
            statistics=["mean", "sum", "count", "std"],
        )
        # Check that the output is complex for mean, sum, std
        self.assertTrue(np.iscomplexobj(ds[f"{DEFAULT_DATA_NAME}_0_mean"].values))
        self.assertTrue(np.iscomplexobj(ds[f"{DEFAULT_DATA_NAME}_0_sum"].values))
        self.assertFalse(np.iscomplexobj(ds[f"{DEFAULT_DATA_NAME}_0_std"].values))
        self.assertTrue(
            np.issubdtype(ds[f"{DEFAULT_DATA_NAME}_0_std"].dtype, np.floating)
        )
        # Count should be real and integer
        self.assertTrue(
            np.issubdtype(ds[f"{DEFAULT_DATA_NAME}_0_count"].dtype, np.integer)
        )
        # Check that the sum is the sum of values in each bin
        self.assertAlmostEqual(
            np.sum(ds[f"{DEFAULT_DATA_NAME}_0_sum"].values), np.sum(values)
        )
        # Check that the mean is the mean of values in each bin
        mask0 = (coords >= -0.2) & (coords < 0.9)
        mask1 = (coords >= 0.9) & (coords < 2.0)
        mask2 = (coords >= 2.0) & (coords <= 3.1)
        means = [
            np.mean(values[mask0]) if np.any(mask0) else 0,
            np.mean(values[mask1]) if np.any(mask1) else 0,
            np.mean(values[mask2]) if np.any(mask2) else 0,
        ]
        np.testing.assert_allclose(
            ds[f"{DEFAULT_DATA_NAME}_0_mean"].values, means, rtol=1e-12
        )

    def test_statistics_complex_min_max_median_raises(self):
        coords = self.coords_1d
        values = np.array(
            [
                1 + 1j,
                2 + 2j,
                3 + 3j,
                4 + 4j,
                5 + 5j,
                6 + 6j,
                7 + 7j,
                8 + 8j,
                9 + 9j,
                10 + 10j,
                11 + 11j,
            ]
        )
        with self.assertRaises(ValueError):
            binned_statistics(
                coords=coords,
                data=values,
                bins=3,
                statistics="min",
            )

        with self.assertRaises(ValueError):
            binned_statistics(
                coords=coords,
                data=values,
                bins=3,
                statistics="max",
            )

        with self.assertRaises(ValueError):
            binned_statistics(
                coords=coords,
                data=values,
                bins=3,
                statistics="median",
            )

    def test_statistics_datetime_coords(self):
        coords = [self.coords_1d, self.date_1d]

        ds = binned_statistics(
            coords=coords,
            data=self.values_1d,
            bins=3,
            dim_names=["x", "time"],
            statistics=["count", "mean"],
        )

        # check if datetime data is handled correctly
        assert ds["time"].dtype.kind == "M"

    def test_statistics_datetime_data_sum(self):
        coords = [self.coords_1d, self.date_1d]
        data = [self.values_1d, self.date_1d]

        with self.assertRaises(ValueError):
            binned_statistics(
                coords=coords,
                data=data,
                bins=3,
                dim_names=["x", "time"],
                statistics=["sum"],  # not supported for datetime data
            )

    def test_statistics_datetime_data(self):
        coords = [self.coords_1d, self.date_1d]
        data = [self.values_1d, self.date_1d]

        ds = binned_statistics(
            coords=coords,
            data=data,
            bins=3,
            dim_names=["x", "time"],
            statistics=["count", "mean", "median", "std", "min", "max"],
        )

        # check if datetime data is handled correctly
        assert ds["time"].dtype.kind == "M"

    def test_statistics_datetime_data_values(self):
        coords = [self.date_1d]
        data = [self.date_1d]

        ds = binned_statistics(
            coords=coords,
            data=data,
            bins=3,
            dim_names=["time"],
            statistics=["count", "mean", "median", "std", "min", "max"],
        )

        # Check that the output is date type
        for var in ds.data_vars:
            if not (var.endswith("count") or var.endswith("std")):
                self.assertTrue(ds[var].dtype.kind == "M")
        self.assertTrue(ds[f"{DEFAULT_DATA_NAME}_0_std"].dtype.kind == "m")

        self.assertIsNone(
            np.testing.assert_array_equal(ds[f"{DEFAULT_DATA_NAME}_0_count"], [4, 3, 4])
        )

        # create mask for three bins
        mask0 = self.date_1d <= self.date_1d[3]
        mask1 = (self.date_1d > self.date_1d[3]) & (self.date_1d < self.date_1d[7])
        mask2 = self.date_1d >= self.date_1d[7]

        self.assertIsNone(
            np.testing.assert_array_equal(
                ds[f"{DEFAULT_DATA_NAME}_0_mean"].values,
                np.array(
                    [
                        "2020-01-02T12:00:00",
                        "2020-01-06T00:00:00",
                        "2020-01-09T12:00:00",
                    ],
                    dtype="datetime64[s]",
                ),
            )
        )

        self.assertIsNone(
            np.testing.assert_array_equal(
                ds[f"{DEFAULT_DATA_NAME}_0_median"].values,
                np.array(
                    [
                        "2020-01-02T12:00:00",
                        "2020-01-06T00:00:00",
                        "2020-01-09T12:00:00",
                    ],
                    dtype="datetime64[s]",
                ),
            )
        )

        self.assertIsNone(
            np.testing.assert_array_equal(
                ds[f"{DEFAULT_DATA_NAME}_0_std"].values,
                np.array(
                    [96598, 70545, 96598],
                    dtype="timedelta64[s]",
                ),
            )
        )

        self.assertIsNone(
            np.testing.assert_array_equal(
                ds[f"{DEFAULT_DATA_NAME}_0_min"].values,
                [
                    min(self.date_1d[mask0]),
                    min(self.date_1d[mask1]),
                    min(self.date_1d[mask2]),
                ],
            )
        )

        self.assertIsNone(
            np.testing.assert_array_equal(
                ds[f"{DEFAULT_DATA_NAME}_0_max"].values,
                [
                    max(self.date_1d[mask0]),
                    max(self.date_1d[mask1]),
                    max(self.date_1d[mask2]),
                ],
            )
        )

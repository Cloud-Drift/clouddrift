import unittest

import numpy as np

from clouddrift.binning import DEFAULT_BINS_NUMBER, histogram


class BinningTests(unittest.TestCase):
    def setUp(self):
        self.coords_1d = np.array(
            [-0.2, 0.3, 0.6, 0.7, 1.2, 1.3, 1.8, 2.1, 2.7, 2.8, 3.1]
        )
        self.bins_range_1d = (0, 3)

        self.coords_2d = [self.coords_1d, self.coords_1d]
        self.bins_range_2d = [(0, 3), (0, 3)]

        self.coords_3d = [self.coords_1d, self.coords_1d, self.coords_1d]
        self.bins_range_3d = [(0, 3), (0, 3), (0, 3)]

    def test_1d_hist_number(self):
        ds = histogram(self.coords_1d, bins=3)

        self.assertEqual(len(ds.dim_0_bin), 3)

    def test_2d_hist_number(self):
        ds = histogram(self.coords_2d, bins=(3, 4))

        self.assertEqual(len(ds.dim_0_bin), 3)
        self.assertEqual(len(ds.dim_1_bin), 4)

    def test_3d_hist_number(self):
        ds = histogram(self.coords_3d, bins=(3, 4, 5))

        self.assertEqual(len(ds.dim_0_bin), 3)
        self.assertEqual(len(ds.dim_1_bin), 4)
        self.assertEqual(len(ds.dim_2_bin), 5)

    def test_hist_center(self):
        for i in range(1, 10):
            ds = histogram(self.coords_1d, bins=i)

            bins_coords = np.linspace(
                np.min(self.coords_1d), np.max(self.coords_1d), i + 1
            )
            bins_center = (bins_coords[:-1] + bins_coords[1:]) / 2

            np.testing.assert_allclose(ds.dim_0_bin.values, bins_center)

    def test_1d_output(self):
        ds = histogram(coords_list=self.coords_1d)
        self.assertEqual(len(ds.dim_0_bin), DEFAULT_BINS_NUMBER)
        self.assertEqual(sum(ds.binned_mean_0.values), len(self.coords_1d))

    def test_1d_output_bins(self):
        n_bins = 3
        ds = histogram(coords_list=self.coords_1d, bins=n_bins)
        self.assertEqual(len(ds.dim_0_bin), n_bins)
        self.assertEqual(sum(ds.binned_mean_0.values), len(self.coords_1d))
        self.assertEqual(len(ds["binned_mean_0"].shape), 1)

    def test_2d_output(self):
        ds = histogram(coords_list=self.coords_2d)
        for v in ds.sizes.values():
            self.assertEqual(v, DEFAULT_BINS_NUMBER)
        self.assertEqual(len(ds["binned_mean_0"].shape), 2)

        n_bins = (3, 4)
        ds = histogram(coords_list=self.coords_2d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(v, n_bins[i])
        self.assertEqual(len(ds["binned_mean_0"].shape), 2)

        n_bins = (3, None)
        ds = histogram(coords_list=self.coords_2d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(
                v, n_bins[i] if n_bins[i] is not None else DEFAULT_BINS_NUMBER
            )
        self.assertEqual(len(ds["binned_mean_0"].shape), 2)

    def test_3d_output(self):
        ds = histogram(coords_list=self.coords_3d)
        for v in ds.sizes.values():
            self.assertEqual(v, DEFAULT_BINS_NUMBER)
        self.assertEqual(len(ds["binned_mean_0"].shape), 3)

        n_bins = (3, 4, 5)
        ds = histogram(coords_list=self.coords_3d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(v, n_bins[i])
        self.assertEqual(len(ds["binned_mean_0"].shape), 3)

        n_bins = (3, 4, None)
        ds = histogram(coords_list=self.coords_3d, bins=n_bins)
        for i, v in enumerate(ds.sizes.values()):
            self.assertEqual(
                v, n_bins[i] if n_bins[i] is not None else DEFAULT_BINS_NUMBER
            )
        self.assertEqual(len(ds["binned_mean_0"].shape), 3)

    def test_hist_range(self):
        ds = histogram(
            coords_list=self.coords_1d, bins=3, bins_range=self.bins_range_1d
        )
        self.assertEqual(
            sum(ds.binned_mean_0.values),
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
        ds = histogram(
            coords_list=self.coords_2d, bins=(3, 3), bins_range=self.bins_range_2d
        )
        self.assertEqual(
            sum(ds.binned_mean_0.values.flatten()),
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
        ds = histogram(
            coords_list=self.coords_3d, bins=(3, 3, 3), bins_range=self.bins_range_3d
        )
        self.assertEqual(
            sum(ds.binned_mean_0.values.flatten()),
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
        ds = histogram(
            coords_list=self.coords_1d,
            bins=4,
            dim_names=["x"],
        )
        self.assertIn("x", ds.sizes)

        ds = histogram(
            coords_list=self.coords_2d,
            bins=(3, 4),
            dim_names=["x", "y"],
        )
        self.assertIn("x", ds.sizes)
        self.assertIn("y", ds.sizes)

    def test_rename_variables(self):
        ds = histogram(
            coords_list=self.coords_1d,
            bins=4,
            new_names=["mean_x"],
        )
        self.assertIn("mean_x", ds.data_vars)

        ds = histogram(
            coords_list=self.coords_2d,
            variables_list=[self.coords_1d, self.coords_1d],
            new_names=["mean_x", "mean_y"],
        )
        self.assertIn("mean_x", ds.data_vars)
        self.assertIn("mean_y", ds.data_vars)

    def test_zeros_to_nan(self):
        ds = histogram(coords_list=self.coords_1d, bins=4, bins_range=(-1, 0))
        empty_bins = ds.binned_mean_0.values == 0

        ds = histogram(
            coords_list=self.coords_1d, bins=4, bins_range=(-1, 0), zeros_to_nan=True
        )
        self.assertTrue(np.isnan(ds.binned_mean_0.values[empty_bins]).all())

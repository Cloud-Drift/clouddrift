import unittest

import numpy as np

from clouddrift.binning import histogram


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

    def test_1d_bins_number(self):
        ds = histogram(self.coords_1d, bins=3)

        self.assertEqual(len(ds.dim_0_bin), 3)

    def test_2d_bins_number(self):
        ds = histogram(self.coords_2d, bins=(3, 4))

        self.assertEqual(len(ds.dim_0_bin), 3)
        self.assertEqual(len(ds.dim_1_bin), 4)

    def test_3d_bins_number(self):
        ds = histogram(self.coords_3d, bins=(3, 4, 5))

        self.assertEqual(len(ds.dim_0_bin), 3)
        self.assertEqual(len(ds.dim_1_bin), 4)
        self.assertEqual(len(ds.dim_2_bin), 5)

    def test_bins_center(self):
        for i in range(1, 10):
            ds = histogram(self.coords_1d, bins=i)

            bins_coords = np.linspace(
                np.min(self.coords_1d), np.max(self.coords_1d), i + 1
            )
            bins_center = (bins_coords[:-1] + bins_coords[1:]) / 2

            np.testing.assert_allclose(ds.dim_0_bin.values, bins_center)

    def test_1d_output(self):
        ds = histogram(coords_list=self.coords_1d)
        self.assertEqual(len(ds.dim_0_bin), 10)
        self.assertEqual(sum(ds.binned_mean_0.values), len(self.coords_1d))

    def test_1d_output_bins(self):
        ds = histogram(coords_list=self.coords_1d, bins=3)
        self.assertEqual(sum(ds.binned_mean_0.values), len(self.coords_1d))

    def test_2d_output(self):
        ds = histogram(
            coords_list=self.coords_2d, bins=3, bins_range=self.bins_range_2d
        )
        raise NotImplementedError("2D binning output test not implemented yet.")

    def test_3d_output(self):
        ds = histogram(
            coords_list=self.coords_3d, bins=3, bins_range=self.bins_range_3d
        )
        raise NotImplementedError("2D binning output test not implemented yet.")

    def test_bins_range(self):
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

    def test_bins_range_2d(self):
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

    def test_bins_range_3d(self):
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

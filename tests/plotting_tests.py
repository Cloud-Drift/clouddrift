import sys
import unittest
from unittest.mock import patch

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from clouddrift.plotting import plot_ragged

if __name__ == "__main__":
    unittest.main()


class plotting_tests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Create trajectories example
        """
        self.lon = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.lat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.rowsize = [3, 3, 4]

    def test_lonlatrowsize(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        lon_test = np.append(self.lon, 3)
        rowsize_test = np.append(self.rowsize, 3)
        with self.assertRaises(ValueError):
            plot_ragged(ax, lon_test, self.lat, self.rowsize)
            plot_ragged(ax, self.lon, self.lat, rowsize_test)

    def test_axis(self):
        ax = 1
        with self.assertRaises(ValueError):
            plot_ragged(ax, self.lon, self.lat, self.rowsize)

    def test_plot_colored_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        l = plot_ragged(
            ax, self.lon, self.lat, self.rowsize, colors=np.arange(len(self.rowsize))
        )
        self.assertIsInstance(l, plt.cm.ScalarMappable)

    def test_plot_colored_datapoints(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        l = plot_ragged(
            ax, self.lon, self.lat, self.rowsize, colors=np.arange(len(self.lat))
        )
        self.assertIsInstance(l, plt.cm.ScalarMappable)

    def test_plot_color_wrong_dimension(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        color_test = np.append(np.arange(len(self.lat)), 3)
        with self.assertRaises(ValueError):
            plot_ragged(ax, self.lon, self.lat, self.rowsize, colors=color_test)

    def test_plot_cartopy_transform(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        with self.assertRaises(ValueError):
            plot_ragged(
                ax,
                self.lon,
                self.lat,
                self.rowsize,
                colors=np.arange(len(self.rowsize)),
            )

    def test_plot_cartopy(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
        l = plot_ragged(
            ax,
            self.lon,
            self.lat,
            self.rowsize,
            colors=np.arange(len(self.rowsize)),
            transform=ccrs.PlateCarree(),
        )
        self.assertIsInstance(l, plt.cm.ScalarMappable)

    def test_plot_segments(self):
        self.lon = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.lat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.rowsize = [3, 3, 4]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
        l = plot_ragged(
            ax,
            self.lon,
            self.lat,
            self.rowsize,
            colors=np.arange(len(self.rowsize)),
            transform=ccrs.PlateCarree(),
        )
        self.assertIsInstance(l, plt.cm.ScalarMappable)

    def test_plot_segments_split(self):
        self.lon = [-170, -175, -180, 175, 170]
        self.lat = [0, 1, 2, 3, 4]
        self.rowsize = [5]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
        l = plot_ragged(
            ax,
            self.lon,
            self.lat,
            self.rowsize,
            colors=np.arange(len(self.rowsize)),
            transform=ccrs.PlateCarree(),
        )
        self.assertIsInstance(l, plt.cm.ScalarMappable)

    def test_plot_segments_split_domain(self):
        self.lon = [-1, -2, -3, 3, 2, 1]
        self.lat = [0, 1, 2, 3, 4, 5]
        self.rowsize = [6]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
        l = plot_ragged(
            ax,
            self.lon,
            self.lat,
            self.rowsize,
            colors=np.arange(len(self.rowsize)),
            transform=ccrs.PlateCarree(),
            tolerance=5,
        )
        self.assertIsInstance(l, plt.cm.ScalarMappable)

    def test_matplotlib_not_installed(self):
        try:
            del sys.modules["clouddrift.plotting"]
        except Exception as e:
            print(f"Could not delete module for testing purposes, error: {e}")

        with patch.dict(sys.modules, {"matplotlib": None}):
            with self.assertRaises(ImportError):
                from clouddrift.plotting import plot_ragged

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                plot_ragged(ax, self.lon, self.lat, self.rowsize)

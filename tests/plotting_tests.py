import numpy as np
import sys
import unittest
from unittest.mock import patch

try:
    import cartopy.crs as ccrs
    from clouddrift.plotting import plot_ragged
    import matplotlib.pyplot as plt

    optional_dependencies_installed = True
except:
    optional_dependencies_installed = False

if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(
    not optional_dependencies_installed,
    "Matplotlib and Cartopy are required for those tests.",
)
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

    def test_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        l = plot_ragged(
            ax, self.lon, self.lat, self.rowsize, colors=np.arange(len(self.rowsize))
        )
        self.assertIsInstance(l, list)

    def test_plot_cartopy_transform(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        with self.assertRaises(ValueError):
            l = plot_ragged(
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
        self.assertIsInstance(l, list)

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
        self.assertIsInstance(l, list)
        self.assertEqual(len(l), 3)

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
        self.assertIsInstance(l, list)
        self.assertEqual(len(l), 2)

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
        self.assertIsInstance(l, list)
        self.assertEqual(len(l), 2)

    def test_matplotlib_not_installed(self):
        del sys.modules["clouddrift.plotting"]
        with patch.dict(sys.modules, {"matplotlib": None}):
            with self.assertRaises(ImportError):
                from clouddrift.plotting import plot_ragged
        # reload for other tests
        from clouddrift.plotting import plot_ragged

    def test_cartopy_not_installed(self):
        del sys.modules["clouddrift.plotting"]
        with patch.dict(sys.modules, {"cartopy": None}):
            from clouddrift.plotting import plot_ragged

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plot_ragged(
                ax,
                self.lon,
                self.lat,
                self.rowsize,
                colors=np.arange(len(self.rowsize)),
            )

        # reload for other tests
        from clouddrift.plotting import plot_ragged

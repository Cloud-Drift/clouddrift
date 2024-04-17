import shutil

import numpy as np

import tests.utils as testutils
from clouddrift.adapters import hurdat2


class hurdat2_integration_tests(testutils.DisableProgressTestCase):
    def test_load_create_ragged_array(self):
        ra = hurdat2.to_raggedarray()
        ds = ra.to_xarray()
        assert "id" in ds.coords
        assert "time" in ds.coords
        assert len(ds.coords["time"]) == len(ra.coords["time"])
        assert len(ds.coords["id"]) == len(ra.coords["id"])

    def test_conversion(self):
        ra = hurdat2.to_raggedarray()
        ra_non_converted = hurdat2.to_raggedarray(convert=False)
        ds = ra.to_xarray()
        ds_non_converted = ra_non_converted.to_xarray()

        assert np.allclose(
            ds["wind_speed"],
            ds_non_converted["wind_speed"] * hurdat2._METERS_IN_NAUTICAL_MILES / 3600,
            equal_nan=True,
        )

        assert np.allclose(
            ds["pressure"], ds_non_converted["pressure"] * 100, equal_nan=True
        )
        assert np.allclose(
            ds["max_sustained_wind_speed_radius"],
            ds_non_converted["max_sustained_wind_speed_radius"]
            * hurdat2._METERS_IN_NAUTICAL_MILES,
            equal_nan=True,
        )

    @classmethod
    def tearDownClass(cls):
        [shutil.rmtree(dir) for dir in [hurdat2._DEFAULT_FILE_PATH]]

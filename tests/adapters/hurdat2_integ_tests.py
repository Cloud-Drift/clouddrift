import logging
import shutil
import unittest

import numpy as np

from clouddrift.adapters import hurdat2, utils

_logger = logging.getLogger(__name__)


class hurdat2_integration_tests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        utils._DEFAULT_SHOW_PROGRESS = False

    def tearDown(self) -> None:
        super().tearDown()
        utils._DEFAULT_SHOW_PROGRESS = True

    def test_load_create_ragged_array(self):
        _logger.info("test hurdat2 adapter, create ragged array")
        ra = hurdat2.to_raggedarray()
        ds = ra.to_xarray()
        assert "id" in ds.coords
        assert "time" in ds.coords
        assert len(ds.coords["time"]) == len(ra.coords["time"])
        assert len(ds.coords["id"]) == len(ra.coords["id"])

    def test_conversion(self):
        _logger.info("test hurdat2 adapter with/without conversion")
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

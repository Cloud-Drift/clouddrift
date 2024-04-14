import logging
import os
import shutil
import unittest

import numpy as np

from clouddrift.adapters import gdp6h, utils

_logger = logging.getLogger(__name__)


class gdp6h_integration_tests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        utils._DEFAULT_SHOW_PROGRESS = False

    def tearDown(self) -> None:
        super().tearDown()
        utils._DEFAULT_SHOW_PROGRESS = True

    def test_load_subset_and_create_aggregate(self):
        _logger.info("test gdp6h adapter, load, subset and create aggregate")
        ra = gdp6h.to_raggedarray(n_random_id=5, tmp_path=gdp6h.GDP_TMP_PATH)
        assert "rowsize" in ra.metadata
        assert "temp" in ra.data
        assert "ve" in ra.data
        assert "vn" in ra.data
        assert ra.coords["id"].dtype == np.int64
        assert len(ra.data["vn"]) == len(ra.coords["time"])
        assert len(ra.metadata["rowsize"]) == len(ra.coords["id"])

        agg_path = os.path.join(gdp6h.GDP_TMP_PATH, "aggregate")
        os.makedirs(agg_path, exist_ok=True)
        ra.to_netcdf(os.path.join(agg_path, "gdp6h_5r_sample.nc"))

    @classmethod
    def tearDownClass(cls):
        [shutil.rmtree(dir) for dir in [gdp6h.GDP_TMP_PATH]]

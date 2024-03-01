import os
import shutil
import unittest

import numpy as np

from clouddrift.adapters import gdp6h


class gdp6h_integration_tests(unittest.TestCase):
    def test_load_subset_and_create_aggregate(self):
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

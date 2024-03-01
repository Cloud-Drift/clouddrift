import os
import shutil
import unittest

import numpy as np

from clouddrift.adapters import gdp1h


class gdp1h_integration_tests(unittest.TestCase):
    def test_load_subset_and_create_aggregate(self):
        test_tasks = [
            (gdp1h.GDP_TMP_PATH, gdp1h.GDP_DATA_URL),
            (gdp1h.GDP_TMP_PATH_EXPERIMENTAL, gdp1h.GDP_DATA_URL_EXPERIMENTAL),
        ]

        for path, url in test_tasks:
            with self.subTest(
                f"test downloading and creating ragged array for: ({url})"
            ):
                ra = gdp1h.to_raggedarray(n_random_id=5, tmp_path=path, url=url)

                assert "rowsize" in ra.metadata
                assert "sst" in ra.data
                assert "ve" in ra.data
                assert "vn" in ra.data
                assert ra.coords["id"].dtype == np.int64
                assert len(ra.data["vn"]) == len(ra.coords["time"])
                assert len(ra.data["ve"]) == len(ra.coords["time"])
                assert len(ra.data["sst"]) == len(ra.coords["time"])
                assert len(ra.metadata["rowsize"]) == len(ra.coords["id"])

                agg_path = os.path.join(gdp1h.GDP_TMP_PATH, "aggregate")
                os.makedirs(agg_path, exist_ok=True)
                ra.to_netcdf(os.path.join(agg_path, "gdp1h_5r_sample.nc"))

    @classmethod
    def tearDownClass(cls):
        [
            shutil.rmtree(dir)
            for dir in [gdp1h.GDP_TMP_PATH, gdp1h.GDP_TMP_PATH_EXPERIMENTAL]
        ]

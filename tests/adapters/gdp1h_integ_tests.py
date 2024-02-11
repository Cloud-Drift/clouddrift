import os
import shutil
import unittest

from clouddrift.adapters import gdp1h


class gdp1h_integration_tests(unittest.TestCase):
    def test_load_subset_and_create_aggregate(self):
        test_tasks = [
            (gdp1h.GDP_TMP_PATH, gdp1h.GDP_DATA_URL), 
            (gdp1h.GDP_TMP_PATH_EXPERIMENTAL, gdp1h.GDP_DATA_URL_EXPERIMENTAL)
        ]

        for (path, url) in test_tasks:
            with self.subTest(f"test downloading and creating ragged array for: ({url})"):
                ra = gdp1h.to_raggedarray(
                    n_random_id=5,
                    tmp_path=path,
                    url=url
                )
                assert 'ID' in ra.metadata
                assert 'rowsize' in ra.metadata
                assert ra.metadata['ID'].size == ra.metadata['rowsize'].size == 5
                agg_path = os.path.join(gdp1h.GDP_TMP_PATH, "aggregate")
                os.makedirs(agg_path, exist_ok=True)
                ra.to_netcdf(os.path.join(agg_path, "gdp1h_5r_sample.nc"))

    @classmethod
    def tearDownClass(cls):
        [shutil.rmtree(dir) for dir in [gdp1h.GDP_TMP_PATH, gdp1h.GDP_TMP_PATH_EXPERIMENTAL]]

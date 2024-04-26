import os
import shutil
import unittest

from clouddrift.adapters import rawfiles


class gdp_rawfiles_integration(unittest.TestCase):
    def test_load_and_create_aggregate(self):
        ra = rawfiles.to_raggedarray()
        # assert "rowsize" in ra.metadata
        # assert "temp" in ra.data
        # assert "ve" in ra.data
        # assert "vn" in ra.data
        # assert ra.coords["id"].dtype == np.int64
        # assert len(ra.data["vn"]) == len(ra.coords["time"])
        # assert len(ra.metadata["rowsize"]) == len(ra.coords["id"])

        agg_path = os.path.join(rawfiles._TMP_PATH, "aggregate")
        os.makedirs(agg_path, exist_ok=True)
        ra.to_netcdf(os.path.join(agg_path, "gdp6h_5r_sample.nc"))

    @classmethod
    def tearDownClass(cls):
        [shutil.rmtree(dir) for dir in [rawfiles._TMP_PATH]]

import shutil
import unittest

from clouddrift.adapters import hurdat2


class hurdat2_integration_tests(unittest.TestCase):
    def test_load_create_ragged_array(self):
        ra = hurdat2.to_raggedarray()
        ds = ra.to_xarray()
        assert "id" in ds.coords
        assert "time" in ds.coords
        assert len(ds.coords["time"]) == len(ra.coords["time"])
        assert len(ds.coords["id"]) == len(ra.coords["id"])

    @classmethod
    def tearDownClass(cls):
        [shutil.rmtree(dir) for dir in [hurdat2._DEFAULT_FILE_PATH]]

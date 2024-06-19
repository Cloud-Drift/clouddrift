import shutil
import unittest

from clouddrift.adapters import gdp_source


class gdp_rawfiles_integration(unittest.TestCase):
    def test_load_and_create_aggregate(self):
        ds = gdp_source.get_dataset(max=1)
        assert ds is not None

    @classmethod
    def tearDownClass(cls):
        [shutil.rmtree(dir) for dir in [gdp_source._TMP_PATH]]

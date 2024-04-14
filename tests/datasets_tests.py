import logging
import unittest

import numpy as np

from clouddrift import datasets
from clouddrift.adapters import utils
from clouddrift.ragged import apply_ragged, subset

if __name__ == "__main__":
    unittest.main()

_logger = logging.getLogger(__name__)


class datasets_tests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        _logger.info("Disabling show progress for all download tasks")
        utils._DEFAULT_SHOW_PROGRESS = False

    def tearDown(self) -> None:
        super().tearDown()
        _logger.info("Disabling show progress for all download tasks")
        utils._DEFAULT_SHOW_PROGRESS = True

    def test_gdp1h(self):
        _logger.info("test gdp1h dataset")
        with datasets.gdp1h() as ds:
            self.assertTrue(ds)

    def test_gdp6h(self):
        _logger.info("test gdp6h dataset")
        with datasets.gdp6h() as ds:
            self.assertTrue(ds)

    def test_glad(self):
        _logger.info("test glad dataset")
        with datasets.glad() as ds:
            self.assertTrue(ds)

    def test_glad_dims_coords(self):
        _logger.info("test glad dataset dim coords")
        with datasets.glad() as ds:
            self.assertTrue(len(ds.sizes) == 2)
            self.assertTrue("obs" in ds.dims)
            self.assertTrue("traj" in ds.dims)
            self.assertTrue(len(ds.coords) == 2)
            self.assertTrue("time" in ds.coords)
            self.assertTrue("id" in ds.coords)

    def test_glad_subset_and_apply_ragged_work(self):
        _logger.info("test glad subset and apply ragged")
        with datasets.glad() as ds:
            ds_sub = subset(
                ds,
                {"id": ["CARTHE_001", "CARTHE_002"]},
                id_var_name="id",
                row_dim_name="traj",
            )
            self.assertTrue(ds_sub)
            mean_lon = apply_ragged(np.mean, [ds_sub.longitude], ds_sub.rowsize)
            self.assertTrue(mean_lon.size == 2)

    def test_spotters_opens(self):
        _logger.info("test spotters dataset")
        with datasets.spotters() as ds:
            self.assertTrue(ds)

    def test_subsurface_floats_opens(self):
        _logger.info("test subsurface floats dataset")
        with datasets.subsurface_floats() as ds:
            self.assertTrue(ds)

    def test_andro_opens(self):
        _logger.info("test andro dataset")
        with datasets.andro() as ds:
            self.assertTrue(ds)

    def test_yomaha_opens(self):
        _logger.info("test yomaha dataset")
        with datasets.yomaha() as ds:
            self.assertTrue(ds)

    def test_mosaic_opens(self):
        _logger.info("test mosaic dataset")
        with datasets.mosaic() as ds:
            self.assertTrue(ds)

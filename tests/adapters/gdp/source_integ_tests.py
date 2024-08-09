import os
import shutil
import unittest

import numpy as np

from clouddrift.adapters import gdp_source
from clouddrift.ragged import unpack


class gdp_source_files_integration(unittest.TestCase):
    def test_load_and_create_aggregate(self):
        """
        Test that each drifters segment has maintianed the row order as its read in
        from the data files. This is tested since the process of generating the
        aggregate dataset is performed in parallel per chunk (default 100_000 rows).

        Each chunk will contain a separate dataset per drifter in the chunk. Since two chunks
        may contain rows for one drifter (chunking can split a drifters trajectory) we join
        these partitioned segments in parallel per drifter.
        """
        ds = gdp_source.to_raggedarray(max=1, chunk_size=1_000, max_chunks=100)
        assert ds is not None

        all_drifter_obs_index = unpack(ds["obs_index"].data, ds["rowsize"])
        for drifter_obs_index in all_drifter_obs_index:
            obs_index_diffs = np.diff(drifter_obs_index, axis=0)
            self.assertEqual(
                np.all(obs_index_diffs == 1).T,
                True,
                "obs_index values not monotonically increasing",
            )

        start_dt_diffs = np.diff(ds["start_date"].data, axis=0)
        self.assertEqual(
            np.all(start_dt_diffs.astype(np.int64) >= 0).T,
            True,
            "Drifter segments not ordered by start date",
        )

        agg_path = os.path.join(gdp_source._TMP_PATH, "aggregate")
        os.makedirs(agg_path, exist_ok=True)
        ds.to_zarr(os.path.join(agg_path, "gdpsource_1f_agg.zarr"))

    @classmethod
    def tearDownClass(cls):
        [shutil.rmtree(dir) for dir in [gdp_source._TMP_PATH]]

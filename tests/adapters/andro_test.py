import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from clouddrift.adapters import andro
from clouddrift.raggedarray import RaggedArray


def _sample_rows() -> str:
    rows = [
        "10.0 20.0 1000 5.0 35.0 1.0 0.1 0.2 0.01 0.02 11.0 21.0 1.1 0.3 0.4 0.03 0.04 12.0 22.0 1.2 0.5 0.6 0.05 0.06 13.0 23.0 1.3 14.0 24.0 1.4 15.0 25.0 1.5 2 1001 1 10",
        "10.5 20.5 1001 5.5 35.1 2.0 0.11 0.21 0.011 0.021 11.5 21.5 2.1 0.31 0.41 0.031 0.041 12.5 22.5 2.2 0.51 0.61 0.051 0.061 13.5 23.5 2.3 14.5 24.5 2.4 15.5 25.5 2.5 3 1001 2 11",
        "30.0 40.0 1002 6.0 34.9 3.0 0.12 0.22 0.012 0.022 31.0 41.0 3.1 0.32 0.42 0.032 0.042 32.0 42.0 3.2 0.52 0.62 0.052 0.062 33.0 43.0 3.3 34.0 44.0 3.4 35.0 45.0 3.5 1 1002 1 20",
    ]
    return "\n".join(rows)


def _build_test_archive() -> bytes:
    payload = BytesIO()
    with ZipFile(payload, mode="w") as archive:
        archive.writestr("data.txt", _sample_rows())
    return payload.getvalue()


class andro_tests(unittest.TestCase):
    def _mock_download(self, download_map, **_kwargs):
        _, output = download_map[0]
        assert isinstance(output, str)
        assert os.path.isdir(os.path.dirname(output))
        with open(output, "wb") as f:
            f.write(_build_test_archive())

    def test_to_raggedarray_builds_expected_structure_and_dtypes(self):
        base = tempfile.mkdtemp()
        tmp_path = os.path.join(base, "andro-cache", "nested")

        with patch(
            "clouddrift.adapters.andro.download_with_progress",
            side_effect=self._mock_download,
        ):
            ra = andro.to_raggedarray(tmp_path=tmp_path, skip_download=False)

        self.assertIsInstance(ra, RaggedArray)
        self.assertTrue(os.path.isdir(tmp_path))

        self.assertTrue(np.array_equal(ra.metadata["rowsize"], np.array([2, 1])))
        self.assertEqual(ra._var_dims["rowsize"], ["traj"])
        self.assertEqual(ra._coord_dims["id"], "traj")
        self.assertEqual(ra._coord_dims["time_d"], "obs")
        self.assertEqual(ra._coord_dims["time_s"], "obs")
        self.assertEqual(ra._coord_dims["time_lp"], "obs")
        self.assertEqual(ra._coord_dims["time_fc"], "obs")
        self.assertEqual(ra._coord_dims["time_lc"], "obs")

        self.assertNotIn("time_d", ra.data)
        self.assertEqual(ra.coords["time_d"].dtype, np.float32)
        self.assertEqual(ra.data["temp_d"].dtype, np.float32)
        self.assertEqual(ra.data["lon_d"].dtype, np.float64)
        self.assertEqual(ra.data["lat_d"].dtype, np.float64)
        self.assertEqual(ra._var_dims["lon_d"], ["obs"])

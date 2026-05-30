import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from clouddrift.adapters import glad
from clouddrift.raggedarray import RaggedArray

# Minimal GLAD CSV: 5 header lines then data rows
_GLAD_CSV = (
    b"% header1\n% header2\n% header3\n% header4\n% header5\n"
    b"CARTHE_001 2012-07-20 18:00:00 29.5 -88.0 100.0 0.10 0.20 0.05\n"
    b"CARTHE_001 2012-07-20 18:15:00 29.6 -87.9 110.0 0.11 0.21 0.06\n"
    b"CARTHE_002 2012-07-20 18:00:00 29.0 -88.5 120.0 0.12 0.22 0.07\n"
)


def _mock_download(requests, **_kwargs):
    """Write _GLAD_CSV into any BytesIO destination."""
    for _, dest, *_ in requests:
        if isinstance(dest, BytesIO):
            dest.write(_GLAD_CSV)


def _make_test_df():
    return pd.DataFrame(
        {
            "id": ["CARTHE_001", "CARTHE_001", "CARTHE_002"],
            "latitude": [29.5, 29.6, 29.0],
            "longitude": [-88.0, -87.9, -88.5],
            "position_error": [100.0, 110.0, 120.0],
            "u": [0.10, 0.11, 0.12],
            "v": [0.20, 0.21, 0.22],
            "velocity_error": [0.05, 0.06, 0.07],
            "obs": pd.to_datetime(
                ["2012-07-20 18:00:00", "2012-07-20 18:15:00", "2012-07-20 18:00:00"]
            ),
        }
    )


class glad_tests(unittest.TestCase):
    def test_get_dataframe_returns_expected_columns(self):
        """get_dataframe returns a DataFrame with the expected columns."""
        # Patch file_size to match the mock payload so the size check passes.
        fake_versions = {"qc2": (glad._DATASET_VERSIONS["qc2"][0], len(_GLAD_CSV))}
        with (
            patch("clouddrift.adapters.glad._DATASET_VERSIONS", fake_versions),
            patch(
                "clouddrift.adapters.glad.download_with_progress",
                side_effect=_mock_download,
            ),
        ):
            df = glad.get_dataframe(version="qc2")

        self.assertEqual(
            list(df.columns),
            ["id", "latitude", "longitude", "position_error", "u", "v", "velocity_error", "obs"],
        )
        self.assertEqual(len(df), 3)

    def test_get_dataframe_raises_on_invalid_version(self):
        """get_dataframe raises ValueError for an unknown version string."""
        with self.assertRaises(ValueError):
            glad.get_dataframe(version="invalid")

    def test_to_raggedarray_returns_raggedarray(self):
        """to_raggedarray returns a RaggedArray instance."""
        with patch("clouddrift.adapters.glad.get_dataframe", return_value=_make_test_df()):
            ra = glad.to_raggedarray()

        self.assertIsInstance(ra, RaggedArray)

    def test_to_raggedarray_dimensions(self):
        """to_raggedarray produces correct traj and obs sizes."""
        with patch("clouddrift.adapters.glad.get_dataframe", return_value=_make_test_df()):
            ds = glad.to_raggedarray().to_xarray()

        self.assertEqual(ds.sizes["traj"], 2)
        self.assertEqual(ds.sizes["obs"], 3)

    def test_to_raggedarray_rowsize(self):
        """to_raggedarray computes per-trajectory rowsize correctly."""
        with patch("clouddrift.adapters.glad.get_dataframe", return_value=_make_test_df()):
            ra = glad.to_raggedarray()

        self.assertTrue(np.array_equal(ra.metadata["rowsize"], [2, 1]))

    def test_to_raggedarray_dtypes(self):
        """to_raggedarray casts float columns to float32 and rowsize to int64."""
        with patch("clouddrift.adapters.glad.get_dataframe", return_value=_make_test_df()):
            ds = glad.to_raggedarray().to_xarray()

        self.assertEqual(ds["latitude"].dtype, np.float32)
        self.assertEqual(ds["longitude"].dtype, np.float32)
        self.assertEqual(ds["rowsize"].dtype, np.int64)

    def test_to_raggedarray_coords(self):
        """to_raggedarray exposes id and time as coordinates."""
        with patch("clouddrift.adapters.glad.get_dataframe", return_value=_make_test_df()):
            ds = glad.to_raggedarray().to_xarray()

        self.assertIn("id", ds.coords)
        self.assertIn("time", ds.coords)

    def test_to_raggedarray_data_vars(self):
        """to_raggedarray includes all expected data variables."""
        with patch("clouddrift.adapters.glad.get_dataframe", return_value=_make_test_df()):
            ds = glad.to_raggedarray().to_xarray()

        for var in [
            "latitude",
            "longitude",
            "position_error",
            "u",
            "v",
            "velocity_error",
            "rowsize",
        ]:
            self.assertIn(var, ds)

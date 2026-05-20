import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from clouddrift.adapters import quicche


class quicche_tests(unittest.TestCase):
    """Unit tests for QUICCHE CARTHE adapter parsing and transformation."""

    def setUp(self):
        """Set up temporary directory for test data."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_data_file(self, filename: str, data_lines: list) -> str:
        """Create a temporary test data file."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, "w") as f:
            for line in data_lines:
                f.write(line + "\n")
        return filepath

    def test_parse_quicche_data_qc2_contains_battery_state(self):
        """QC2 parsing keeps battery_state and omits flag."""
        data_lines = [
            "1865211416	0-4435604	Q1_0001	2023-03-15T10:00:00.000Z	1679048400	25.73166	-80.1633	UNLIMITED-TRACK	GOOD",
            "1865211417	0-4435605	Q1_0002	2023-03-15T10:05:00.000Z	1679048700	25.73200	-80.1640	UNLIMITED-TRACK	GOOD",
        ]
        filepath = self._create_test_data_file("test_qc2.dat", data_lines)

        df = quicche._parse_quicche_data(filepath, "qc2")

        self.assertEqual(len(df), 2)
        self.assertEqual(
            list(df.columns),
            ["drifter_id", "time", "latitude", "longitude", "battery_state"],
        )
        self.assertEqual(df.iloc[0]["drifter_id"], "Q1_0001")
        self.assertEqual(df.iloc[0]["latitude"], 25.73166)
        self.assertEqual(df.iloc[0]["longitude"], -80.1633)
        self.assertEqual(df.iloc[0]["battery_state"], "GOOD")
        self.assertTrue(pd.notna(df.iloc[0]["time"]))
        self.assertNotIn("flag", df.columns)

    def test_parse_quicche_data_qc1_contains_flag(self):
        """QC1 parsing keeps battery_state and flag, with empty flag when missing."""
        data_lines = [
            "1865211416	0-4435604	Q1_0001	2023-03-15T10:00:00.000Z	1679048400	25.73166	-80.1633	UNLIMITED-TRACK	GOOD	PRE",
            "1865211418	0-4435606	Q1_0002	2023-03-15T10:05:00.000Z	1679048700	25.73200	-80.1640	UNLIMITED-TRACK	LOW	BAD_POS",
            "1865211417	0-4435605	Q1_0002	2023-03-15T10:05:00.000Z	1679048700	25.73200	-80.1640	UNLIMITED-TRACK	GOOD",
        ]
        filepath = self._create_test_data_file("test_mixed.dat", data_lines)

        df = quicche._parse_quicche_data(filepath, "qc1")

        self.assertEqual(len(df), 3)
        self.assertIn("battery_state", df.columns)
        self.assertIn("flag", df.columns)
        self.assertEqual(df.iloc[0]["drifter_id"], "Q1_0001")
        self.assertIn("PRE", set(df["flag"].values))
        self.assertIn("BAD_POS", set(df["flag"].values))
        self.assertIn("", set(df["flag"].values))

    def test_parse_sorted_by_drifter_id_and_time(self):
        """Test that parsed data is sorted by drifter_id and time."""
        data_lines = [
            "1	1	Q1_0002	2023-03-15T10:10:00.000Z	1	25.7	-80.2	A	GOOD",
            "2	2	Q1_0001	2023-03-15T10:00:00.000Z	1	25.7	-80.2	A	GOOD",
            "3	3	Q1_0001	2023-03-15T10:05:00.000Z	1	25.7	-80.2	A	GOOD",
        ]
        filepath = self._create_test_data_file("test_unsorted.dat", data_lines)

        df = quicche._parse_quicche_data(filepath, "qc2")

        # Check sort order: Q1_0001 entries first, then Q1_0002
        self.assertEqual(df.iloc[0]["drifter_id"], "Q1_0001")
        self.assertEqual(df.iloc[1]["drifter_id"], "Q1_0001")
        self.assertEqual(df.iloc[2]["drifter_id"], "Q1_0002")

    def test_dataframe_to_ragged_xarray_dimensions(self):
        """Test ragged array conversion produces correct dimensions."""
        df = pd.DataFrame(
            {
                "drifter_id": ["Q1_0001", "Q1_0001", "Q1_0002", "Q1_0002", "Q1_0002"],
                "time": pd.date_range("2023-03-15", periods=5, freq="5min"),
                "latitude": [25.73, 25.74, 25.80, 25.81, 25.82],
                "longitude": [-80.16, -80.17, -80.20, -80.21, -80.22],
            }
        )

        ds = quicche._dataframe_to_ragged_xarray(df, "qc3")

        # Check dimensions
        self.assertIn("traj", ds.dims)
        self.assertIn("obs", ds.dims)
        self.assertEqual(ds.sizes["traj"], 2)  # Two drifters
        self.assertEqual(ds.sizes["obs"], 5)  # Five observations

    def test_dataframe_to_ragged_xarray_coordinates(self):
        """Test ragged array has correct coordinates and data variables."""
        df = pd.DataFrame(
            {
                "drifter_id": ["Q1_0001", "Q1_0001", "Q1_0002"],
                "time": pd.date_range("2023-03-15", periods=3, freq="5min"),
                "latitude": [25.73, 25.74, 25.80],
                "longitude": [-80.16, -80.17, -80.20],
            }
        )

        ds = quicche._dataframe_to_ragged_xarray(df, "qc2")

        # Check coordinates
        self.assertIn("id", ds.coords)
        self.assertIn("time", ds.coords)
        self.assertNotIn("index", ds.coords)
        self.assertNotIn("index", ds.variables)

        # Check data variables
        self.assertIn("latitude", ds.data_vars)
        self.assertIn("longitude", ds.data_vars)
        self.assertIn("rowsize", ds.data_vars)

    def test_qc1_ragged_includes_flag_and_battery_state(self):
        """QC1 ragged dataset exposes both battery_state and flag variables."""
        df = pd.DataFrame(
            {
                "drifter_id": ["Q1_0001", "Q1_0001", "Q1_0002"],
                "time": pd.date_range("2023-03-15", periods=3, freq="5min"),
                "latitude": [25.73, 25.74, 25.80],
                "longitude": [-80.16, -80.17, -80.20],
                "battery_state": ["GOOD", "LOW", "GOOD"],
                "flag": ["PRE", "", "BAD_POS"],
            }
        )

        ds = quicche._dataframe_to_ragged_xarray(df, "qc1")
        self.assertIn("battery_state", ds.data_vars)
        self.assertIn("flag", ds.data_vars)

    def test_qc3_ragged_excludes_flag_and_battery_state(self):
        """QC3 ragged dataset should not include battery_state or flag variables."""
        df = pd.DataFrame(
            {
                "drifter_id": ["Q1_0001", "Q1_0001"],
                "time": pd.date_range("2023-03-15", periods=2, freq="5min"),
                "latitude": [25.73, 25.74],
                "longitude": [-80.16, -80.17],
            }
        )

        ds = quicche._dataframe_to_ragged_xarray(df, "qc3")
        self.assertNotIn("battery_state", ds.data_vars)
        self.assertNotIn("flag", ds.data_vars)

    def test_dataframe_to_ragged_xarray_rowsize(self):
        """Test rowsize is computed correctly."""
        df = pd.DataFrame(
            {
                "drifter_id": ["Q1_0001", "Q1_0001", "Q1_0002", "Q1_0002", "Q1_0002"],
                "time": pd.date_range("2023-03-15", periods=5, freq="5min"),
                "latitude": [25.73, 25.74, 25.80, 25.81, 25.82],
                "longitude": [-80.16, -80.17, -80.20, -80.21, -80.22],
            }
        )

        ds = quicche._dataframe_to_ragged_xarray(df, "qc3")

        rowsize = ds["rowsize"].values
        # Q1_0001 has 2 observations, Q1_0002 has 3
        self.assertTrue(np.array_equal(rowsize, [2, 3]))

    def test_dataframe_to_ragged_xarray_dtypes(self):
        """Test data types are correctly cast to project conventions."""
        df = pd.DataFrame(
            {
                "drifter_id": ["Q1_0001", "Q1_0001"],
                "time": pd.date_range("2023-03-15", periods=2, freq="5min"),
                "latitude": [25.73, 25.74],
                "longitude": [-80.16, -80.17],
            }
        )

        ds = quicche._dataframe_to_ragged_xarray(df, "qc3")

        # Check float32 for lat/lon
        self.assertEqual(ds["latitude"].dtype, np.float32)
        self.assertEqual(ds["longitude"].dtype, np.float32)

        # Check int64 for rowsize
        self.assertEqual(ds["rowsize"].dtype, np.int64)

    def test_dataframe_to_ragged_xarray_global_attrs(self):
        """Test global attributes are set with correct version info."""
        df = pd.DataFrame(
            {
                "drifter_id": ["Q1_0001"],
                "time": pd.date_range("2023-03-15", periods=1, freq="5min"),
                "latitude": [25.73],
                "longitude": [-80.16],
            }
        )

        ds_raw = quicche._dataframe_to_ragged_xarray(df, "raw")
        ds_qc1 = quicche._dataframe_to_ragged_xarray(df, "qc1")
        ds_qc2 = quicche._dataframe_to_ragged_xarray(df, "qc2")
        ds_qc3 = quicche._dataframe_to_ragged_xarray(df, "qc3")

        # Check version is in title and attributes
        self.assertIn("RAW", ds_raw.attrs["title"])
        self.assertIn("QC1", ds_qc1.attrs["title"])
        self.assertIn("QC2", ds_qc2.attrs["title"])
        self.assertIn("QC3", ds_qc3.attrs["title"])
        self.assertEqual(ds_raw.attrs["qc_level"], "raw")
        self.assertEqual(ds_qc1.attrs["qc_level"], "qc1")
        self.assertEqual(ds_qc2.attrs["qc_level"], "qc2")
        self.assertEqual(ds_qc3.attrs["qc_level"], "qc3")

    def test_to_xarray_version_validation(self):
        """Test that invalid versions raise ValueError."""
        with self.assertRaises(ValueError):
            quicche.to_xarray(version="invalid", tmp_path=self.test_dir)

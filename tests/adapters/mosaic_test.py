import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from clouddrift.adapters import mosaic
from clouddrift.raggedarray import RaggedArray


def _make_test_dataframes():
    obs_df = pd.DataFrame(
        {
            "latitude": [89.5, 89.6, 89.4, 89.3, 89.2],
            "longitude": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            name="datetime",
        ),
    )
    traj_df = pd.DataFrame(
        {
            "rowsize": [3, 2],
            "Deployment Leg": [1, 2],
            "Buoy Type": ["IMB", "SIMBA"],
            "Deployment Date": ["2019-10-04", "2019-10-15"],
            "Deployment Datetime": ["2019-10-04T12:00:00", "2019-10-15T09:00:00"],
            "First Data Datetime": ["2020-01-01T00:00:00", "2020-01-04T00:00:00"],
            "Last Data Datetime": ["2020-01-03T00:00:00", "2020-01-05T00:00:00"],
        },
        index=pd.Index(["SENSOR_001", "SENSOR_002"], name="Sensor ID"),
    )
    return obs_df, traj_df


class mosaic_tests(unittest.TestCase):
    def test_to_raggedarray_returns_raggedarray(self):
        """to_raggedarray returns a RaggedArray instance."""
        with patch(
            "clouddrift.adapters.mosaic.get_dataframes",
            return_value=_make_test_dataframes(),
        ):
            ra = mosaic.to_raggedarray()

        self.assertIsInstance(ra, RaggedArray)

    def test_to_raggedarray_dimensions(self):
        """to_raggedarray produces correct traj and obs sizes."""
        with patch(
            "clouddrift.adapters.mosaic.get_dataframes",
            return_value=_make_test_dataframes(),
        ):
            ds = mosaic.to_raggedarray().to_xarray()

        self.assertEqual(ds.sizes["traj"], 2)
        self.assertEqual(ds.sizes["obs"], 5)

    def test_to_raggedarray_rowsize(self):
        """to_raggedarray computes per-trajectory rowsize correctly."""
        with patch(
            "clouddrift.adapters.mosaic.get_dataframes",
            return_value=_make_test_dataframes(),
        ):
            ra = mosaic.to_raggedarray()

        self.assertTrue(np.array_equal(ra.metadata["rowsize"], [3, 2]))

    def test_to_raggedarray_coords(self):
        """to_raggedarray exposes id (Sensor ID) and time as coordinates."""
        with patch(
            "clouddrift.adapters.mosaic.get_dataframes",
            return_value=_make_test_dataframes(),
        ):
            ds = mosaic.to_raggedarray().to_xarray()

        self.assertIn("id", ds.coords)
        self.assertIn("time", ds.coords)
        self.assertTrue(np.array_equal(ds["id"].values, ["SENSOR_001", "SENSOR_002"]))

    def test_to_raggedarray_traj_metadata(self):
        """to_raggedarray includes all traj-level sensor columns as metadata."""
        with patch(
            "clouddrift.adapters.mosaic.get_dataframes",
            return_value=_make_test_dataframes(),
        ):
            ds = mosaic.to_raggedarray().to_xarray()

        self.assertIn("Deployment Leg", ds)
        self.assertIn("Buoy Type", ds)
        self.assertEqual(ds["Deployment Leg"].dims, ("traj",))

    def test_to_raggedarray_datetime_columns_converted(self):
        """to_raggedarray converts string datetime traj columns to datetime64."""
        with patch(
            "clouddrift.adapters.mosaic.get_dataframes",
            return_value=_make_test_dataframes(),
        ):
            ds = mosaic.to_raggedarray().to_xarray()

        self.assertTrue(np.issubdtype(ds["Deployment Date"].dtype, np.datetime64))

    def test_to_raggedarray_data_vars(self):
        """to_raggedarray includes latitude and longitude as obs-level data."""
        with patch(
            "clouddrift.adapters.mosaic.get_dataframes",
            return_value=_make_test_dataframes(),
        ):
            ds = mosaic.to_raggedarray().to_xarray()

        self.assertIn("latitude", ds)
        self.assertIn("longitude", ds)
        self.assertEqual(ds["latitude"].dims, ("obs",))

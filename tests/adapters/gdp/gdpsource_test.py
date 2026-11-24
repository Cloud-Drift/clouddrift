import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import xarray as xr

from clouddrift.adapters.gdp import gdpsource
from clouddrift.raggedarray import RaggedArray


def _make_drifter_ds(id_: int, obs_indices: list[int], start_date: str, lat_offset: float = 0.0):
    """Build a synthetic per-drifter xr.Dataset matching the structure returned by _parallel_get."""
    n_obs = len(obs_indices)
    return xr.Dataset(
        {
            "rowsize": (["traj"], np.array([n_obs], dtype=np.int64)),
            "wmo_number": (["traj"], np.array([id_ * 10], dtype=np.int64)),
            "program_number": (["traj"], np.array([1], dtype=np.int64)),
            "buoys_type": (["traj"], np.array(["SVPB"])),
            "start_date": (["traj"], np.array([start_date], dtype="datetime64[ns]")),
            "start_lat": (["traj"], np.array([lat_offset], dtype=np.float64)),
            "start_lon": (["traj"], np.array([0.0], dtype=np.float64)),
            "end_date": (["traj"], np.array([start_date], dtype="datetime64[ns]")),
            "end_lat": (["traj"], np.array([lat_offset], dtype=np.float64)),
            "end_lon": (["traj"], np.array([0.0], dtype=np.float64)),
            "drogue_off_date": (["traj"], np.array(["1970-01-01"], dtype="datetime64[ns]")),
            "death_code": (["traj"], np.array([0], dtype=np.int64)),
            "latitude": (["obs"], np.full(n_obs, lat_offset, dtype=np.float32)),
            "longitude": (["obs"], np.zeros(n_obs, dtype=np.float32)),
            "position_datetime": (
                ["obs"],
                np.array(["2020-01-01"] * n_obs, dtype="datetime64[ns]"),
            ),
            "sensor_datetime": (
                ["obs"],
                np.array(["2020-01-01"] * n_obs, dtype="datetime64[ns]"),
            ),
            "drogue": (["obs"], np.ones(n_obs, dtype=np.float32)),
            "sst": (["obs"], np.full(n_obs, 25.0, dtype=np.float32)),
            "voltage": (["obs"], np.full(n_obs, 3.2, dtype=np.float32)),
            "sensor4": (["obs"], np.zeros(n_obs, dtype=np.float32)),
            "sensor5": (["obs"], np.zeros(n_obs, dtype=np.float32)),
            "sensor6": (["obs"], np.zeros(n_obs, dtype=np.float32)),
            "qualityIndex": (["obs"], np.ones(n_obs, dtype=np.float32)),
        },
        coords={
            "id": (["traj"], np.array([id_], dtype=np.int64)),
            "obs_index": (["obs"], np.array(obs_indices, dtype=np.int32)),
        },
    )


_MOCK_DRIFTER_DATASETS = [
    _make_drifter_ds(1001, [0, 1, 2], "2020-01-01", lat_offset=10.0),
    _make_drifter_ds(1002, [3, 4], "2020-06-01", lat_offset=20.0),
]


class gdpsource_tests(unittest.TestCase):
    def _call_to_raggedarray(self, **kwargs):
        with (
            patch("clouddrift.adapters.gdp.gdpsource.download_with_progress"),
            patch("clouddrift.adapters.gdp.gdpsource.get_gdp_metadata"),
            patch(
                "clouddrift.adapters.gdp.gdpsource._parallel_get",
                new=AsyncMock(return_value=_MOCK_DRIFTER_DATASETS),
            ),
        ):
            return gdpsource.to_raggedarray(skip_download=True, **kwargs)

    def test_to_raggedarray_returns_raggedarray(self):
        """to_raggedarray returns a RaggedArray instance."""
        ra = self._call_to_raggedarray()
        self.assertIsInstance(ra, RaggedArray)

    def test_to_raggedarray_traj_count(self):
        """to_raggedarray produces one trajectory per drifter."""
        ra = self._call_to_raggedarray()
        self.assertEqual(len(ra.coords["id"]), 2)

    def test_to_raggedarray_obs_count(self):
        """to_raggedarray produces the correct total number of observations."""
        ra = self._call_to_raggedarray()
        self.assertEqual(len(ra.coords["obs_index"]), 5)

    def test_to_raggedarray_rowsize(self):
        """to_raggedarray computes per-trajectory rowsize correctly."""
        ra = self._call_to_raggedarray()
        np.testing.assert_array_equal(ra.metadata["rowsize"], [3, 2])

    def test_to_raggedarray_sorted_by_start_date(self):
        """to_raggedarray sorts trajectories by start date (ascending)."""
        ra = self._call_to_raggedarray()
        np.testing.assert_array_equal(ra.coords["id"], [1001, 1002])

    def test_to_raggedarray_coords_present(self):
        """to_raggedarray includes id and obs_index as coordinates."""
        ra = self._call_to_raggedarray()
        self.assertIn("id", ra.coords)
        self.assertIn("obs_index", ra.coords)

    def test_to_raggedarray_data_vars_present(self):
        """to_raggedarray includes all expected data variables."""
        ra = self._call_to_raggedarray()
        for var in gdpsource._DATA_VARS:
            self.assertIn(var, ra.data)

    def test_to_raggedarray_metadata_vars_present(self):
        """to_raggedarray includes all expected metadata variables."""
        ra = self._call_to_raggedarray()
        for var in gdpsource._METADATA_VARS:
            self.assertIn(var, ra.metadata)

    def test_to_raggedarray_attrs_global(self):
        """to_raggedarray attaches global attributes."""
        ra = self._call_to_raggedarray()
        self.assertIn("title", ra.attrs_global)

    def test_to_raggedarray_to_xarray_round_trip(self):
        """to_xarray() on the result is an xr.Dataset with correct dimensions."""
        ra = self._call_to_raggedarray()
        ds = ra.to_xarray()
        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual(ds.sizes["traj"], 2)
        self.assertEqual(ds.sizes["obs"], 5)

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from clouddrift.adapters import subsurface_floats
from clouddrift.raggedarray import RaggedArray

# Unix epoch day number (1970-01-01) used by the subsurface floats adapter
_ORIGIN_DATENUM = 719529


def _make_mat_data():
    """Return a minimal mock scipy.io.loadmat output for 2 floats.

    Float 1001 has 3 observations; float 1002 has 2 observations.
    Each float belongs to its own experiment (EXP1 / EXP2).
    """

    def scalar_cells(*values):
        arr = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            arr[i] = np.array([v])
        return arr

    def array_cells(*arrays):
        arr = np.empty(len(arrays), dtype=object)
        for i, a in enumerate(arrays):
            arr[i] = np.array(a, dtype=float)
        return arr

    return {
        "expList": scalar_cells("EXP1", "EXP2"),
        "expName": scalar_cells("Experiment 1", "Experiment 2"),
        "expOrg": scalar_cells("WHOI", "AOML"),
        "expPI": scalar_cells("Smith", "Jones"),
        "fltType": scalar_cells("APEX", "SOLO"),
        "indexExp": scalar_cells(1, 2),
        "indexFlt": scalar_cells(1001, 1002),
        "dtnum": array_cells(
            [_ORIGIN_DATENUM + 1.0, _ORIGIN_DATENUM + 2.0, _ORIGIN_DATENUM + 3.0],
            [_ORIGIN_DATENUM + 4.0, _ORIGIN_DATENUM + 5.0],
        ),
        "lon": array_cells([-10.0, -10.1, -10.2], [-20.0, -20.1]),
        "lat": array_cells([45.0, 45.1, 45.2], [50.0, 50.1]),
        "p": array_cells([100.0, 101.0, 102.0], [200.0, 201.0]),
        "t": array_cells([10.0, 10.1, 10.2], [5.0, 5.1]),
        "u": array_cells([0.1, 0.2, 0.3], [0.4, 0.5]),
        "v": array_cells([0.2, 0.3, 0.4], [0.5, 0.6]),
    }


def _mock_download(requests, **_kwargs):
    """Write a single non-empty byte so the file-size check passes."""
    for _, dest, *_ in requests:
        if isinstance(dest, str):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(b"\x00")


class subsurface_floats_tests(unittest.TestCase):
    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def _run_to_raggedarray(self):
        with (
            patch(
                "clouddrift.adapters.subsurface_floats.download_with_progress",
                side_effect=_mock_download,
            ),
            patch(
                "clouddrift.adapters.subsurface_floats.scipy.io.loadmat",
                return_value=_make_mat_data(),
            ),
            patch(
                "clouddrift.adapters.subsurface_floats.os.path.getsize",
                return_value=1,
            ),
        ):
            return subsurface_floats.to_raggedarray(tmp_path=self.tmp_path)

    def test_to_raggedarray_returns_raggedarray(self):
        """to_raggedarray returns a RaggedArray instance."""
        ra = self._run_to_raggedarray()
        self.assertIsInstance(ra, RaggedArray)

    def test_to_raggedarray_dimensions(self):
        """to_raggedarray produces correct traj and obs sizes."""
        ds = self._run_to_raggedarray().to_xarray()
        self.assertEqual(ds.sizes["traj"], 2)
        self.assertEqual(ds.sizes["obs"], 5)

    def test_to_raggedarray_rowsize(self):
        """to_raggedarray computes per-trajectory rowsize correctly."""
        ra = self._run_to_raggedarray()
        self.assertTrue(np.array_equal(ra.metadata["rowsize"], [3, 2]))

    def test_to_raggedarray_coords(self):
        """to_raggedarray exposes id and time as coordinates."""
        ds = self._run_to_raggedarray().to_xarray()
        self.assertIn("id", ds.coords)
        self.assertIn("time", ds.coords)

    def test_to_raggedarray_dtypes(self):
        """to_raggedarray keeps lat/lon as float64 and casts others to float32."""
        ds = self._run_to_raggedarray().to_xarray()
        self.assertEqual(ds["lat"].dtype, np.float64)
        self.assertEqual(ds["lon"].dtype, np.float64)
        self.assertEqual(ds["pres"].dtype, np.float32)
        self.assertEqual(ds["temp"].dtype, np.float32)
        self.assertEqual(ds["ve"].dtype, np.float32)
        self.assertEqual(ds["vn"].dtype, np.float32)

    def test_to_raggedarray_traj_metadata(self):
        """to_raggedarray includes expected traj-level metadata variables."""
        ds = self._run_to_raggedarray().to_xarray()
        for var in ["expList", "expName", "expOrg", "expPI", "fltType", "indexExp"]:
            self.assertIn(var, ds)
            self.assertEqual(ds[var].dims, ("traj",))

    def test_to_raggedarray_raises_on_empty_file(self):
        """to_raggedarray raises ConnectionError when the downloaded file is empty."""
        with (
            patch(
                "clouddrift.adapters.subsurface_floats.download_with_progress",
                side_effect=_mock_download,
            ),
            patch(
                "clouddrift.adapters.subsurface_floats.os.path.getsize",
                return_value=0,
            ),
        ):
            with self.assertRaises(ConnectionError):
                subsurface_floats.to_raggedarray(tmp_path=self.tmp_path)

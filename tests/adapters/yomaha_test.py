import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from clouddrift.adapters import yomaha
from clouddrift.raggedarray import RaggedArray

# -------------------------------------------------------------------------
# Minimal test-file content
# -------------------------------------------------------------------------

# float_types.txt: "float_type_id: float_type" rows + 4 footer lines (skipfooter=4)
_FLOAT_TYPES_TXT = "1: APEX\n2: SOLO\n0: METOCEAN\nfooter1\nfooter2\nfooter3\nfooter4\n"

# DACs.txt: "dac_id: dac_name"
_DACS_TXT = "1: AOML\n2: CORIOLIS\n"

# 0-date_time.txt: last update date string
_DATE_TIME_TXT = "2024-01-15\n"

# WMO2DAC2type.txt: "id wmo_id dac_id float_type_id" (whitespace-separated)
_WMO2DAC2TYPE_TXT = "1001 9001 1 1\n1002 9002 1 2\n"

# end-prog.lst: not read by to_raggedarray, just needs to exist
_END_PROG_LST = ""

# yomaha07.dat: 28 whitespace-separated columns per row (2 floats × 2 cycles)
# Columns: lon_d lat_d pres_d time_d ve_d vn_d err_ve_d err_vn_d
#          lon_s lat_s time_s ve_s vn_s err_ve_s err_vn_s
#          lon_lp lat_lp time_lp lon_fc lat_fc time_fc lon_lc lat_lc time_lc
#          surf_fix id cycle time_inv
_YOMAHA_DAT = (
    "180.0 45.0 1000.0 1000.0 10.0 10.0 1.0 1.0 "
    "181.0 45.5 1001.0 5.0 5.0 0.5 0.5 "
    "179.0 44.5 999.0 180.0 45.0 1000.0 181.0 45.5 1001.0 "
    "3.0 1001.0 1.0 0.0\n"
    "180.1 45.1 1000.0 1010.0 10.0 10.0 1.0 1.0 "
    "181.1 45.6 1011.0 5.0 5.0 0.5 0.5 "
    "180.0 45.0 1009.0 180.1 45.1 1010.0 181.1 45.6 1011.0 "
    "3.0 1001.0 2.0 0.0\n"
    "200.0 50.0 1000.0 1000.0 8.0 8.0 1.0 1.0 "
    "201.0 50.5 1001.0 4.0 4.0 0.5 0.5 "
    "199.0 49.5 999.0 200.0 50.0 1000.0 201.0 50.5 1001.0 "
    "2.0 1002.0 1.0 0.0\n"
    "200.1 50.1 1000.0 1010.0 8.0 8.0 1.0 1.0 "
    "201.1 50.6 1011.0 4.0 4.0 0.5 0.5 "
    "200.0 50.0 1009.0 200.1 50.1 1010.0 201.1 50.6 1011.0 "
    "2.0 1002.0 2.0 0.0\n"
)

_FILE_CONTENTS = {
    "float_types.txt": _FLOAT_TYPES_TXT,
    "DACs.txt": _DACS_TXT,
    "0-date_time.txt": _DATE_TIME_TXT,
    "WMO2DAC2type.txt": _WMO2DAC2TYPE_TXT,
    "end-prog.lst": _END_PROG_LST,
    "yomaha07.dat": _YOMAHA_DAT,
}


def _create_test_files(tmp_path: str):
    for filename, content in _FILE_CONTENTS.items():
        with open(os.path.join(tmp_path, filename), "w") as f:
            f.write(content)


def _noop_download(*_args, **_kwargs):
    """No-op mock for download_with_progress; files are pre-created."""
    pass


class yomaha_tests(unittest.TestCase):
    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()
        _create_test_files(self.tmp_path)

    def _run_to_raggedarray(self):
        with patch(
            "clouddrift.adapters.yomaha.download_with_progress",
            side_effect=_noop_download,
        ):
            return yomaha.to_raggedarray(tmp_path=self.tmp_path, skip_download=True)

    def test_to_raggedarray_returns_raggedarray(self):
        """to_raggedarray returns a RaggedArray instance."""
        ra = self._run_to_raggedarray()
        self.assertIsInstance(ra, RaggedArray)

    def test_to_raggedarray_dimensions(self):
        """to_raggedarray produces correct traj and obs sizes."""
        ds = self._run_to_raggedarray().to_xarray()
        self.assertEqual(ds.sizes["traj"], 2)
        self.assertEqual(ds.sizes["obs"], 4)

    def test_to_raggedarray_rowsize(self):
        """to_raggedarray computes per-trajectory rowsize correctly."""
        ra = self._run_to_raggedarray()
        self.assertTrue(np.array_equal(ra.metadata["rowsize"], [2, 2]))

    def test_to_raggedarray_time_coords(self):
        """to_raggedarray exposes time_d, time_s, time_lp, time_fc, time_lc as coordinates."""
        ds = self._run_to_raggedarray().to_xarray()
        for tc in ["time_d", "time_s", "time_lp", "time_fc", "time_lc"]:
            self.assertIn(tc, ds.coords)

    def test_to_raggedarray_traj_metadata(self):
        """to_raggedarray includes wmo_id, dac_id, float_type as traj-level metadata."""
        ds = self._run_to_raggedarray().to_xarray()
        for var in ["wmo_id", "dac_id", "float_type"]:
            self.assertIn(var, ds)
            self.assertEqual(ds[var].dims, ("traj",))

    def test_to_raggedarray_dtypes(self):
        """to_raggedarray keeps lat/lon pairs as float64 and casts others to float32."""
        ds = self._run_to_raggedarray().to_xarray()
        for var in ["lat_d", "lon_d", "lat_s", "lon_s", "lat_lp", "lon_lp", "lat_lc", "lon_lc"]:
            self.assertEqual(ds[var].dtype, np.float64, msg=f"{var} should be float64")
        for var in ["ve_d", "vn_d", "pres_d"]:
            self.assertEqual(ds[var].dtype, np.float32, msg=f"{var} should be float32")

    def test_to_raggedarray_id_coord(self):
        """to_raggedarray exposes id as a traj-level coordinate."""
        ds = self._run_to_raggedarray().to_xarray()
        self.assertIn("id", ds.coords)
        self.assertTrue(np.array_equal(ds["id"].values, [1001, 1002]))

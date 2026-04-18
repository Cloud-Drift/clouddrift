import numpy as np

import tests.utils as testutils
from clouddrift import adapters, datasets
from clouddrift.ragged import apply_ragged, subset


class datasets_tests(testutils.DisableProgressTestCase):
    def test_gdp1h(self):
        with datasets.gdp1h() as ds:
            self.assertTrue(ds)

    def test_gdp6h(self):
        with datasets.gdp6h() as ds:
            self.assertTrue(ds)

    def test_ibtracs(self):
        """
        The original dataset contains data points for entire time span of the dataset. Due to this, we trim
        the original data array to only contain up to the numobs associated to the storm. In this test we validate
        the trimmed data and ensure the left over is all NaN.
        """
        options = dict(version="v04r01", kind="LAST_3_YEARS")
        ragged_ds = adapters.ibtracs.to_raggedarray(**options)
        ds = adapters.ibtracs._get_original_dataset(**options)
        ragged_ds_first = subset(
            ragged_ds, {"storm": 0}, row_dim_name="storm", rowsize_var_name="numobs"
        )
        ds_first = ds.sel(storm=0)
        first_numobs: np.ndarray = ds_first.numobs.data
        first_numobs = first_numobs.astype(int)

        # Both transformed and original data array should contain the same values
        self.assertTrue(
            np.all(
                np.allclose(
                    ds_first.usa_r34.data[:first_numobs],
                    ragged_ds_first.usa_r34.data,
                    equal_nan=True,
                )
            )
        )

        # The rest of the values should be nan
        self.assertTrue(
            np.all(
                np.isnan(
                    ds_first.usa_r34.data[first_numobs + 1 :],
                )
            )
        )

    def test_glad(self):
        with datasets.glad() as ds:
            self.assertTrue(ds)

    def test_glad_dims_coords(self):
        with datasets.glad() as ds:
            self.assertTrue(len(ds.sizes) == 2)
            self.assertTrue("obs" in ds.dims)
            self.assertTrue("traj" in ds.dims)
            self.assertTrue(len(ds.coords) == 2)
            self.assertTrue("time" in ds.coords)
            self.assertTrue("id" in ds.coords)

    def test_glad_subset_and_apply_ragged_work(self):
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
        with datasets.spotters() as ds:
            self.assertTrue(ds)

    def test_subsurface_floats_opens(self):
        with datasets.subsurface_floats() as ds:
            self.assertTrue(ds)

    def test_andro_opens(self):
        with datasets.andro() as ds:
            self.assertTrue(ds is not None)
            self.assertTrue(len(ds.variables) > 0)
            self.assertTrue(len(ds["lon_d"]) > 0)

            self.assertTrue(
                len(ds.lat_d[np.logical_or(ds.lat_d > 90, ds.lat_d < -90)]) == 0
            )
            self.assertTrue(
                len(ds.lat_d[np.logical_or(ds.lon_d > 180, ds.lon_d < -180)]) == 0
            )

    def test_yomaha_opens(self):
        with datasets.yomaha() as ds:
            self.assertTrue(ds)

    def test_mosaic_opens(self):
        with datasets.mosaic() as ds:
            self.assertTrue(ds)

    def test_cape_basin_opens(self):
        with datasets.cape_basin() as ds:
            self.assertTrue(ds is not None)
            self.assertTrue(len(ds.variables) > 0)
            self.assertTrue("latitude" in ds.variables)
            self.assertTrue("longitude" in ds.variables)

    def test_cape_basin_default_version(self):
        """Test that cape_basin defaults to qc3."""
        with datasets.cape_basin() as ds:
            self.assertEqual(ds.attrs.get("qc_level"), "qc3")

    def test_cape_basin_qc2_version(self):
        """Test cape_basin with explicit qc2 version."""
        with datasets.cape_basin(version="qc2") as ds:
            self.assertEqual(ds.attrs.get("qc_level"), "qc2")

    def test_cape_basin_dims_coords(self):
        """Test cape_basin has expected ragged array structure."""
        with datasets.cape_basin() as ds:
            self.assertIn("traj", ds.dims)
            self.assertIn("obs", ds.dims)
            self.assertIn("id", ds.coords)
            self.assertIn("time", ds.coords)
            self.assertNotIn("index", ds.coords)
            self.assertNotIn("index", ds.variables)

    def test_cape_basin_decode_times_false(self):
        """Test cape_basin respects decode_times parameter."""
        ds_decoded = datasets.cape_basin(decode_times=True)
        ds_not_decoded = datasets.cape_basin(decode_times=False)

        # Decoded version should have datetime64
        self.assertTrue(np.issubdtype(ds_decoded["time"].dtype, np.datetime64))
        # Not decoded version can be various numeric types
        self.assertFalse(np.issubdtype(ds_not_decoded["time"].dtype, np.datetime64))

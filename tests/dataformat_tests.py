import unittest
from unittest import TestCase
import os
import xarray as xr
import numpy as np
from clouddrift import dataformat
import awkward._v2 as ak

if __name__ == "__main__":
    unittest.main()


class dataformat_tests(TestCase):
    def setUp(self):
        """
        Create ragged array and output netCDF and Parquet file
        """
        self.drifter_id = [1, 2, 3]
        self.rowsize = [10, 8, 2]
        self.nb_obs = np.sum(self.rowsize)
        self.nb_traj = len(self.drifter_id)
        self.attrs_global = {
            "title": "test trajectories",
            "history": "version xyz",
        }
        self.variables_coords = {"ids": "ids", "time": "t", "lon": "longitude", "lat": "latitude"}

        # append xr.Dataset to a list
        list_ds = []
        for i in range(0, len(self.rowsize)):
            xr_coords = {}
            for var in ["longitude", "latitude", "t"]:
                xr_coords[var] = (
                    ["obs"],
                    np.random.rand(self.rowsize[i]),
                    {"long_name": f"variable {var}", "units": "-"},
                )
            xr_coords["ids"] = (
                ["obs"],
                np.ones(self.rowsize[i], dtype="int") * self.drifter_id[i],
                {"long_name": f"variable ids", "units": "-"},
            )

            xr_data = {}
            xr_data["ID"] = (
                ["traj"],
                [self.drifter_id[i]],
                {"long_name": f"variable ID", "units": "-"},
            )
            xr_data["rowsize"] = (
                ["traj"],
                [self.rowsize[i]],
                {"long_name": f"variable ID", "units": "-"},
            )
            xr_data["temp"] = (
                ["obs"],
                np.random.rand(self.rowsize[i]),
                {"long_name": f"variable temp", "units": "-"},
            )

            list_ds.append(
                xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=self.attrs_global)
            )

        # create test ragged array
        self.ra = dataformat.create_ragged_array(
            [0, 1, 2], lambda i: list_ds[i], self.variables_coords, ["ID", "rowsize"], ["temp"]
        )

        # output archive
        self.ra.to_netcdf("test_archive.nc")
        self.ra.to_parquet("test_archive.parquet")

    def tearDown(self):
        """
        Clean up saved archives
        """
        os.remove("test_archive.nc")
        os.remove("test_archive.parquet")

    def test_length_ragged_arrays(self):
        """
        Validate the size of the ragged array variables
        """
        for var in ["lon", "lat", "time", "ids"]:
            self.assertEqual(len(self.ra.coords[var]), self.nb_obs)
        self.assertEqual(len(self.ra.metadata["ID"]), self.nb_traj)
        self.assertEqual(len(self.ra.data["temp"]), self.nb_obs)

    def test_rename_coords(self):
        """
        Validate that coordinates name were modified
        """
        for key in self.ra.coords:
            self.assertTrue(key in ("ids", "lon", "lat", "time"))

    def test_variable_attrs(self):
        """
        Validate the variable attributes are properly transfered to the ragged array object.
        Note: as part of this test `long_name` is variable but `units` are always "-"
        """
        for var in ["lon", "lat", "time"]:  # coords are rename but not attributes here
            self.assertEqual(
                self.ra.attrs_variables[var]["long_name"],
                f"variable {self.variables_coords[var]}",
            )
            self.assertEqual(self.ra.attrs_variables[var]["units"], "-")

        for var in ["ids", "ID", "temp"]:
            self.assertEqual(
                self.ra.attrs_variables[var]["long_name"], f"variable {var}"
            )
            self.assertEqual(self.ra.attrs_variables[var]["units"], "-")

    def test_global_attrs(self):
        """
        Validate the global attributes are properly transfered to the ragged array object
        """
        self.assertEqual(self.ra.attrs_global, self.attrs_global)

    def compare_awkward_array(self, ds):
        """
        Compare the returned Awkward Array after initializing from a netCDF/Parquet archive
        with the ragged array object
        """
        # dimensions
        self.assertEqual(self.ra.nb_obs, len(ak.flatten(ds.obs["temp"])))
        self.assertEqual(self.ra.nb_traj, len(ds.obs["temp"]))

        # coords
        for var in ["lon", "lat", "time", "ids"]:
            self.assertTrue(
                np.allclose(self.ra.coords[var], ak.flatten(ds.obs[var]).to_numpy())
            )
            self.assertTrue(
                self.ra.attrs_variables[var] == ds.obs[var].layout.parameters["attrs"]
            )

        # metadata and
        self.assertTrue(np.allclose(self.ra.metadata["ID"], ds["ID"].to_numpy()))
        self.assertTrue(
            self.ra.attrs_variables["ID"] == ds["ID"].layout.parameters["attrs"]
        )

        # data
        self.assertTrue(
            np.allclose(self.ra.data["temp"], ak.flatten(ds.obs["temp"]).to_numpy())
        )
        self.assertTrue(
            self.ra.attrs_variables["temp"] == ds.obs["temp"].layout.parameters["attrs"]
        )

    def test_netcdf_output(self):
        """
        Validate the netCDF output archive
        """
        ds = dataformat.read_from_netcdf("test_archive.nc")
        self.compare_awkward_array(ds)

    def test_parquet_output(self):
        """
        Validate the netCDF output archive
        """
        ds = dataformat.read_from_parquet("test_archive.parquet")
        self.compare_awkward_array(ds)
import os
import unittest
from unittest import TestCase

import awkward as ak  # type: ignore
import numpy as np
import xarray as xr

from clouddrift import RaggedArray

NETCDF_ARCHIVE = "test_archive.nc"
PARQUET_ARCHIVE = "test_archive.parquet"

if __name__ == "__main__":
    unittest.main()


class raggedarray_tests(TestCase):
    @classmethod
    def setUpClass(self):
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
        self.variables_coords = [("id", "traj"), ("time", "obs")]


        # append xr.Dataset to a list
        list_ds = []
        for i in range(0, len(self.rowsize)):
            xr_coords = {}
            xr_coords["id"] = (
                ["traj"],
                [self.drifter_id[i]],
                {"long_name": "variable id", "units": "-"},
            )

            xr_coords["time"] = (
                ["obs"],
                np.ones(self.rowsize[i], dtype="int") * self.drifter_id[i],
                {"long_name": "variable time", "units": "-"},
            )

            xr_data = {}
            xr_data["rowsize"] = (
                ["traj"],
                [self.rowsize[i]],
                {"long_name": "variable rowsize", "units": "-"},
            )
            xr_data["temp"] = (
                ["obs"],
                np.random.rand(self.rowsize[i]),
                {"long_name": "variable temp", "units": "-"},
            )

            list_ds.append(
                xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=self.attrs_global)
            )

        # create test ragged array
        self.name_coords=["id", "time"]
        self.name_meta=["rowsize"]
        self.name_data=["temp"]
        self.name_dims={"traj": "rows", "obs": "obs"}
        self.coord_dims={"id": "traj", "time": "obs"}
        self.ra = RaggedArray.from_files(
            [0, 1, 2],
            lambda i: list_ds[i],
            self.name_coords,
            self.name_meta,
            self.name_data,
            self.name_dims,
            lambda i: self.rowsize[i]
        )

        # output archive
        self.ra.to_netcdf(NETCDF_ARCHIVE)
        self.ra.to_parquet(PARQUET_ARCHIVE)

    @classmethod
    def tearDownClass(self):
        """
        Clean up saved archives
        """
        os.remove(NETCDF_ARCHIVE)
        os.remove(PARQUET_ARCHIVE)

    def test_from_awkward(self):
        ra = RaggedArray.from_awkward(
            ak.from_parquet(PARQUET_ARCHIVE), 
            self.name_coords,
            self.name_dims,
            self.coord_dims
        )
        self.compare_awkward_array(ra.to_awkward())

    def test_from_xarray(self):
        ra = RaggedArray.from_xarray(xr.open_dataset(NETCDF_ARCHIVE), "traj")
        self.compare_awkward_array(ra.to_awkward())

    def test_from_xarray_dim_names(self):
        ds = xr.open_dataset("test_archive.nc")
        
        ra = RaggedArray.from_xarray(
            ds.rename_dims({"traj": "t", "obs": "o"}),
            rows_dim_name="t",
            obs_dim_name="o",
        )
        self.compare_awkward_array(ra.to_awkward())

    def test_length_ragged_arrays(self):
        """
        Validate the size of the ragged array variables
        """
        self.assertEqual(len(self.ra.coords["id"]), self.nb_traj)
        self.assertEqual(len(self.ra.coords["time"]), self.nb_obs)

        self.assertEqual(len(self.ra.metadata["rowsize"]), self.nb_traj)
        self.assertEqual(len(self.ra.data["temp"]), self.nb_obs)

    def test_variable_attrs(self):
        """
        Validate the variable attributes are properly transferred to the ragged array object.
        Note: as part of this test `long_name` is variable but `units` are always "-"
        """
        for var in ["id", "time", "rowsize", "temp"]:
            self.assertEqual(
                self.ra.attrs_variables[var]["long_name"],
                f"variable {var}",
            )
            self.assertEqual(self.ra.attrs_variables[var]["units"], "-")

    def test_global_attrs(self):
        """
        Validate the global attributes are properly transferred to the ragged array object
        """
        self.assertEqual(self.ra.attrs_global, self.attrs_global)

    def compare_awkward_array(self, ds):
        """
        Compare the returned Awkward Array after initializing from a netCDF/Parquet archive
        with the ragged array object
        """
        # dimensions
        self.assertEqual(len(self.ra.data["temp"]), len(ak.flatten(ds.obs["temp"])))

        # coords
        for var in ["time", "id"]:
            if var == "time":
                self.assertTrue(
                    np.allclose(self.ra.coords[var], ak.flatten(ds.obs[var]).to_numpy())
                )
            else:
                self.assertTrue(
                    np.allclose(self.ra.coords[var], ds.obs[var].to_numpy())
                )
            self.assertTrue(
                self.ra.attrs_variables[var] == ds.obs[var].layout.parameters["attrs"]
            )

        # metadata
        self.assertTrue(
            np.allclose(self.ra.metadata["rowsize"], ds["rowsize"].to_numpy())
        )
        self.assertTrue(
            self.ra.attrs_variables["rowsize"]
            == ds["rowsize"].layout.parameters["attrs"]
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
        ds = RaggedArray.from_netcdf(NETCDF_ARCHIVE, "traj")
        self.compare_awkward_array(ds.to_awkward())

    def test_parquet_output(self):
        """
        Validate the netCDF output archive
        """
        ds = RaggedArray.from_parquet(
            PARQUET_ARCHIVE, 
            self.name_coords,
            self.name_dims,
            self.coord_dims
        )
        self.compare_awkward_array(ds.to_awkward())

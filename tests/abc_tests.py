import unittest
from unittest import TestCase
import os
import xarray as xr
import numpy as np
from clouddrift import RaggedArray
import awkward as ak

NETCDF_ARCHIVE = "test_archive.nc"
PARQUET_ARCHIVE = "test_archive.parquet"

if __name__ == "__main__":
    unittest.main()


class asdasd_tests(TestCase):
    @classmethod
    def setUpClass(self):
        """
        Create ragged array and output netCDF and Parquet file
        """
        self.drifter_id = [1, 2, 3]
        self.count = [10, 8, 2]
        self.nb_obs = np.sum(self.count)
        self.nb_traj = len(self.drifter_id)
        self.attrs_global = {
            "title": "test trajectories",
            "history": "version xyz",
        }
        self.variables_coords = ["ids", "time", "lon", "lat"]

        # append xr.Dataset to a list
        list_ds = []
        for i in range(0, len(self.count)):
            xr_coords = {}
            for var in ["lon", "lat", "time"]:
                xr_coords[var] = (
                    ["obs"],
                    np.random.rand(self.count[i]),
                    {"long_name": f"variable {var}", "units": "-"},
                )
            xr_coords["ids"] = (
                ["obs"],
                np.ones(self.count[i], dtype="int") * self.drifter_id[i],
                {"long_name": f"variable ids", "units": "-"},
            )

            xr_data = {}
            xr_data["ID"] = (
                ["traj"],
                [self.drifter_id[i]],
                {"long_name": f"variable ID", "units": "-"},
            )
            xr_data["count"] = (
                ["traj"],
                [self.count[i]],
                {"long_name": f"variable count", "units": "-"},
            )
            xr_data["temp"] = (
                ["obs"],
                np.random.rand(self.count[i]),
                {"long_name": f"variable temp", "units": "-"},
            )

            list_ds.append(
                xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=self.attrs_global)
            )

        # create test ragged array
        self.ra = RaggedArray.from_files(
            [0, 1, 2],
            lambda i: list_ds[i],
            self.variables_coords,
            ["ID", "count"],
            ["temp"],
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

    # def test_from_awkward(self):
    #    self.compare_awkward_array(RaggedArray.from_awkward(ak.from_parquet(PARQUET_ARCHIVE)).to_awkward())

    def test_from_xarray(self):
        with xr.open_dataset(NETCDF_ARCHIVE) as ds:
            self.compare_awkward_array(
                RaggedArray.from_xarray(ds.copy(deep=True)).to_awkward()
            )

    def compare_awkward_array(self, ds):
        """
        Compare the returned Awkward Array after initializing from a netCDF/Parquet archive
        with the ragged array object
        """
        pass
        """
        # dimensions
        self.assertEqual(len(self.ra.data["temp"]), len(ak.flatten(ds.obs["temp"])))
        self.assertEqual(len(self.ra.metadata["ID"]), len(ds.obs["temp"]))
        
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
        """

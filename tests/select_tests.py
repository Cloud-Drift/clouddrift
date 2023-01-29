import unittest
from unittest import TestCase
import numpy as np
import xarray as xr
import awkward as ak
from clouddrift import RaggedArray, select

if __name__ == "__main__":
    unittest.main()


def is_ak_empty(ds: ak.Array):
    if len(ds) == 0 and not ds.fields:
        return True
    else:
        return False


class select_tests(TestCase):
    def setUp(self):
        """
        Create ragged array and output netCDF and Parquet file
        """
        drifter_id = [1, 2, 3]
        rowsize = [5, 4, 2]
        longitude = [[-121, -111, 51, 61, 71], [12, 22, 32, 42], [103, 113]]
        latitude = [[-90, -45, 45, 90, 0], [10, 20, 30, 40], [10, 20]]
        t = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [4, 5]]
        ids = [[1, 1, 1, 1, 1], [2, 2, 2, 2], [3, 3]]
        test = [
            [True, True, True, False, False],
            [True, False, False, False],
            [False, False],
        ]
        nb_obs = np.sum(rowsize)
        nb_traj = len(drifter_id)
        attrs_global = {
            "title": "test trajectories",
            "history": "version xyz",
        }
        variables_coords = ["ids", "time", "lon", "lat"]

        coords = {"lon": longitude, "lat": latitude, "ids": ids, "time": t}
        metadata = {"ID": drifter_id, "rowsize": rowsize}
        data = {"test": test}

        # append xr.Dataset to a list
        list_ds = []
        for i in range(0, len(rowsize)):
            xr_coords = {}
            for var in coords.keys():
                xr_coords[var] = (
                    ["obs"],
                    coords[var][i],
                    {"long_name": f"variable {var}", "units": "-"},
                )

            xr_data = {}
            for var in metadata.keys():
                xr_data[var] = (
                    ["traj"],
                    [metadata[var][i]],
                    {"long_name": f"variable {var}", "units": "-"},
                )

            for var in data.keys():
                xr_data[var] = (
                    ["obs"],
                    data[var][i],
                    {"long_name": f"variable {var}", "units": "-"},
                )

            list_ds.append(
                xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=attrs_global)
            )

        # create test ragged array
        ra = RaggedArray.from_files(
            [0, 1, 2],
            lambda i: list_ds[i],
            variables_coords,
            ["ID", "rowsize"],
            ["test"],
        )

        self.ds = ra.to_awkward()

    def test_equal(self):
        ds_sub = select.subset(self.ds, {"test": True})
        self.assertEqual(len(ds_sub.ID), 2)

    def test_select(self):
        ds_sub = select.subset(self.ds, {"ID": [1, 2]})
        self.assertTrue(ak.all(ds_sub.ID == ak.Array([1, 2])))
        self.assertEqual(len(ds_sub.obs.lon), 2)

    def test_range(self):
        # positive
        ds_sub = select.subset(self.ds, {"lon": (0, 180)})
        self.assertTrue(ak.all(ds_sub.obs.lon[0] == ak.Array([51, 61, 71])))
        self.assertTrue(ak.all(ds_sub.obs.lon[1] == ak.Array([12, 22, 32, 42])))
        self.assertTrue(ak.all(ds_sub.obs.lon[2] == ak.Array([103, 113])))

        # negative range
        ds_sub = select.subset(self.ds, {"lon": (-180, 0)})
        self.assertEqual(len(ds_sub.ID), 1)
        self.assertEqual(ds_sub.ID[0], 1)
        self.assertEqual(len(ds_sub.obs.lon), 1)
        self.assertTrue(ak.all(ds_sub.obs.lon[0] == ak.Array([-121, -111])))

        # both
        ds_sub = select.subset(self.ds, {"lon": (-30, 30)})
        self.assertEqual(len(ds_sub.ID), 1)
        self.assertEqual(ds_sub.ID[0], 2)
        self.assertEqual(len(ds_sub.obs.lon), 1)
        self.assertTrue(ak.all(ds_sub.obs.lon[0] == ak.Array([12, 22])))

    def test_combine(self):
        ds_sub = select.subset(
            self.ds, {"ID": [1, 2], "lat": (-90, 20), "lon": (-180, 25), "test": True}
        )
        self.assertTrue(ak.all(ds_sub.ID == ak.Array([1, 2])))
        self.assertTrue(ak.all(ds_sub.obs.lon == ak.Array([[-121, -111], [12]])))
        self.assertTrue(ak.all(ds_sub.obs.lat == ak.Array([[-90, -45], [10]])))

    def test_empty(self):
        ds_sub = select.subset(self.ds, {"ID": 3, "lon": (-180, 0)})
        self.assertTrue(is_ak_empty(ds_sub))

    def test_unknown_var(self):
        with self.assertRaises(ValueError):
            select.subset(self.ds, {"a": 10})

        with self.assertRaises(ValueError):
            select.subset(self.ds, {"lon": (0, 180), "a": (0, 10)})

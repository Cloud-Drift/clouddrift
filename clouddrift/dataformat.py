import awkward._v2 as ak
import xarray as xr
import numpy as np
from collections.abc import Callable
from typing import Tuple, Optional
from tqdm import tqdm


class ragged_array:
    def __init__(
        self,
        coords: dict,
        metadata: dict,
        data: dict,
        attrs_global: Optional[dict] = {},
        attrs_variables: Optional[dict] = {},
    ):
        self.coords = coords
        self.metadata = metadata
        self.data = data
        self.attrs_global = attrs_global
        self.attrs_variables = attrs_variables
        self.validate_attributes()

    @classmethod
    def from_files(
        cls,
        indices: list,
        preprocess_func: Callable[[int], xr.Dataset],
        vars_coords: dict,
        vars_meta: list = [],
        vars_data: list = [],
        rowsize_func: Optional[Callable[[int], int]] = None,
    ):
        # if no method is supplied, get the dimension from the preprocessing function
        rowsize_func = (
            rowsize_func if rowsize_func else lambda i: preprocess_func(i).dims["obs"]
        )
        rowsize = cls.number_of_observations(rowsize_func, indices)
        coords, metadata, data = cls.allocate(
            preprocess_func, indices, rowsize, vars_coords, vars_meta, vars_data
        )
        attrs_global, attrs_variables = cls.attributes(
            preprocess_func(indices[0]), vars_coords, vars_meta, vars_data
        )

        return cls(coords, metadata, data, attrs_global, attrs_variables)

    @classmethod
    def from_netcdf(cls, filename: str):
        """"""
        coords = {}
        metadata = {}
        data = {}
        attrs_global = {}
        attrs_variables = {}

        with xr.open_dataset(filename) as ds:
            nb_traj = ds.dims["traj"]
            nb_obs = ds.dims["obs"]

            attrs_global = ds.attrs

            for var in ds.coords.keys():
                coords[var] = ds[var].data
                attrs_variables[var] = ds[var].attrs

            for var in ds.data_vars.keys():
                if len(ds[var]) == nb_traj:
                    metadata[var] = ds[var].data
                elif len(ds[var]) == nb_obs:
                    data[var] = ds[var].data
                else:
                    print(
                        f"Error: variable '{var}' has unknown dimension size of {len(ds[var])}, which is not traj={nb_traj} or obs={nb_obs}."
                    )
                attrs_variables[var] = ds[var].attrs

        return cls(coords, metadata, data, attrs_global, attrs_variables)

    @classmethod
    def from_parquet(cls, filename: str):
        """"""
        coords = {}
        metadata = {}
        data = {}
        attrs_global = {}
        attrs_variables = {}

        ds = ak.from_parquet(filename)
        attrs_global = ds.layout.parameters["attrs"]

        name_coords = ["time", "lon", "lat", "ids"]
        for var in name_coords:
            coords[var] = ak.flatten(ds.obs[var]).to_numpy()
            attrs_variables[var] = ds.obs[var].layout.parameters["attrs"]

        for var in [v for v in ds.fields if v != "obs"]:
            metadata[var] = ds[var].to_numpy()
            attrs_variables[var] = ds[var].layout.parameters["attrs"]

        for var in [v for v in ds.obs.fields if v not in name_coords]:
            data[var] = ak.flatten(ds.obs[var]).to_numpy()
            attrs_variables[var] = ds.obs[var].layout.parameters["attrs"]

        return cls(coords, metadata, data, attrs_global, attrs_variables)

    @staticmethod
    def number_of_observations(
        rowsize_func: Callable[[int], int], indices: list
    ) -> np.array:
        """
        Load files and get the size of the observations.
        """
        rowsize = np.zeros(len(indices), dtype="int")

        for i, index in tqdm(
            enumerate(indices),
            total=len(indices),
            desc="Retrieving the number of obs",
            ncols=80,
        ):
            rowsize[i] = rowsize_func(index)
        return rowsize

    @staticmethod
    def attributes(
        ds: xr.Dataset, vars_coords: dict, vars_meta: list, vars_data: list
    ) -> Tuple[dict, dict]:
        attrs_global = ds.attrs

        attrs_variables = {}

        # coordinates
        for var in vars_coords.keys():
            attrs_variables[var] = ds[vars_coords[var]].attrs

        # metadata and data
        for var in vars_meta + vars_data:
            attrs_variables[var] = ds[var].attrs

        return attrs_global, attrs_variables

    @staticmethod
    def allocate(
        preprocess_func: Callable[[int], xr.Dataset],
        indices: list,
        rowsize: list,
        vars_coords: dict,
        vars_meta: list,
        vars_data: list,
    ) -> Tuple[dict, dict, dict]:
        """
        Allocate and fill for the ragged array associated with all variables
        """

        # open one file to get dtype of variables
        ds = preprocess_func(indices[0])
        nb_traj = len(rowsize)
        nb_obs = np.sum(rowsize).astype("int")
        index_traj = np.insert(np.cumsum(rowsize), 0, 0)

        # allocate memory
        coords = {}
        for var in vars_coords.keys():
            coords[var] = np.zeros(nb_obs, dtype=ds[vars_coords[var]].dtype)

        metadata = {}
        for var in vars_meta:
            metadata[var] = np.zeros(nb_traj, dtype=ds[var].dtype)

        data = {}
        for var in vars_data:
            data[var] = np.zeros(nb_obs, dtype=ds[var].dtype)
        ds.close()

        # loop and fill the ragged array
        for i, index in tqdm(
            enumerate(indices),
            total=len(indices),
            desc="Filling the Ragged Array",
            ncols=80,
        ):
            with preprocess_func(index) as ds:
                size = rowsize[i]
                oid = index_traj[i]

                for var in vars_coords.keys():
                    coords[var][oid : oid + size] = ds[vars_coords[var]].data

                for var in vars_meta:
                    metadata[var][i] = ds[var][0].data

                for var in vars_data:
                    data[var][oid : oid + size] = ds[var].data

        return coords, metadata, data

    def validate_attributes(self):
        """
        If not specify it creates an empty attributes for each variable
        """
        for key in (
            list(self.coords.keys())
            + list(self.metadata.keys())
            + list(self.data.keys())
        ):
            if key not in self.attrs_variables:
                self.attrs_variables[key] = {}

    def to_xarray(self):
        """
        Output the ragged array dataformat to an xr.Dataset
        """
        xr_coords = {}
        for var in self.coords.keys():
            xr_coords[var] = (["obs"], self.coords[var], self.attrs_variables[var])

        xr_data = {}
        for var in self.metadata.keys():
            xr_data[var] = (["traj"], self.metadata[var], self.attrs_variables[var])

        for var in self.data.keys():
            xr_data[var] = (["obs"], self.data[var], self.attrs_variables[var])

        return xr.Dataset(coords=xr_coords, data_vars=xr_data, attrs=self.attrs_global)

    def to_awkward(self):
        """
        Output the ragged array dataformat to an Awkward Array archive
        """
        index_traj = np.insert(np.cumsum(self.metadata["rowsize"]), 0, 0)
        offset = ak.index.Index64(index_traj)

        data = []
        for var in self.coords.keys():
            data.append(
                ak.contents.ListOffsetArray(
                    offset,
                    ak.contents.NumpyArray(self.coords[var]),
                    parameters={"attrs": self.attrs_variables[var]},
                )
            )
        for var in self.data.keys():
            data.append(
                ak.contents.ListOffsetArray(
                    offset,
                    ak.contents.NumpyArray(self.data[var]),
                    parameters={"attrs": self.attrs_variables[var]},
                )
            )
        data_names = list(self.coords.keys()) + list(self.data.keys())

        metadata = []
        for var in self.metadata.keys():
            metadata.append(
                ak.with_parameter(
                    self.metadata[var],
                    "attrs",
                    self.attrs_variables[var],
                    highlevel=False,
                )
            )
        metadata_names = list(self.metadata.keys())

        # include the data inside the metadata list as a nested array
        metadata_names.append("obs")
        metadata.append(ak.Array(ak.contents.RecordArray(data, data_names)).layout)

        return ak.Array(
            ak.contents.RecordArray(
                metadata, metadata_names, parameters={"attrs": self.attrs_global}
            )
        )

    def to_netcdf(self, filename: str):
        """
        Export ragged array dataset to NetCDF archive

        Args: filename: path of the archive
        """
        self.to_xarray().to_netcdf(filename)
        return

    def to_parquet(self, filename: str):
        """
        Export ragged array dataset to parquet archive

        Args: filename: path of the archive
        """
        ak.to_parquet(self.to_awkward(), filename)
        return

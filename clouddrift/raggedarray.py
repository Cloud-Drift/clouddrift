"""
This module defines the RaggedArray class, which is the intermediate data
structure used by CloudDrift to process custom Lagrangian datasets to Xarray
Datasets and Awkward Arrays.
"""
import awkward as ak
from clouddrift.ragged import rowsize_to_index
import xarray as xr
import numpy as np
from collections.abc import Callable
from typing import Tuple, Optional
from tqdm import tqdm
import warnings


class RaggedArray:
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
    def from_awkward(
        cls,
        array: ak.Array,
        name_coords: Optional[list] = ["time", "lon", "lat", "ids"],
    ):
        """Load a RaggedArray instance from an Awkward Array.

        Parameters
        ----------
        array : ak.Array
            Awkward Array instance to load the data from
        name_coords : list, optional
            Names of the coordinate variables in the ragged arrays

        Returns
        -------
        RaggedArray
            A RaggedArray instance
        """
        coords = {}
        metadata = {}
        data = {}
        attrs_variables = {}

        attrs_global = array.layout.parameters["attrs"]

        for var in name_coords:
            coords[var] = ak.flatten(array.obs[var]).to_numpy()
            attrs_variables[var] = array.obs[var].layout.parameters["attrs"]

        for var in [v for v in array.fields if v != "obs"]:
            metadata[var] = array[var].to_numpy()
            attrs_variables[var] = array[var].layout.parameters["attrs"]

        for var in [v for v in array.obs.fields if v not in name_coords]:
            data[var] = ak.flatten(array.obs[var]).to_numpy()
            attrs_variables[var] = array.obs[var].layout.parameters["attrs"]

        return cls(coords, metadata, data, attrs_global, attrs_variables)

    @classmethod
    def from_files(
        cls,
        indices: list,
        preprocess_func: Callable[[int], xr.Dataset],
        name_coords: list,
        name_meta: Optional[list] = [],
        name_data: Optional[list] = [],
        rowsize_func: Optional[Callable[[int], int]] = None,
        **kwargs,
    ):
        """Generate a ragged array archive from a list of trajectory files

        Parameters
        ----------
        indices : list
            Identification numbers list to iterate
        preprocess_func : Callable[[int], xr.Dataset]
            Returns a processed xarray Dataset from an identification number
        name_coords : list
            Name of the coordinate variables to include in the archive
        name_meta : list, optional
            Name of metadata variables to include in the archive (Defaults to [])
        name_data : list, optional
            Name of the data variables to include in the archive (Defaults to [])
        rowsize_func : Optional[Callable[[int], int]], optional
            Returns the number of observations from an identification number (to speed up processing) (Defaults to None)

        Returns
        -------
        RaggedArray
            A RaggedArray instance
        """
        # if no method is supplied, get the dimension from the preprocessing function
        rowsize_func = (
            rowsize_func
            if rowsize_func
            else lambda i, **kwargs: preprocess_func(i, **kwargs).dims["obs"]
        )
        rowsize = cls.number_of_observations(rowsize_func, indices, **kwargs)
        coords, metadata, data = cls.allocate(
            preprocess_func,
            indices,
            rowsize,
            name_coords,
            name_meta,
            name_data,
            **kwargs,
        )
        attrs_global, attrs_variables = cls.attributes(
            preprocess_func(indices[0], **kwargs),
            name_coords,
            name_meta,
            name_data,
        )

        return cls(coords, metadata, data, attrs_global, attrs_variables)

    @classmethod
    def from_netcdf(cls, filename: str):
        """Read a ragged arrays archive from a NetCDF file.

        This is a thin wrapper around ``from_xarray()``.

        Parameters
        ----------
        filename : str
            File name of the NetCDF archive to read.

        Returns
        -------
        RaggedArray
            A ragged array instance
        """
        return cls.from_xarray(xr.open_dataset(filename))

    @classmethod
    def from_parquet(
        cls, filename: str, name_coords: Optional[list] = ["time", "lon", "lat", "ids"]
    ):
        """Read a ragged array from a parquet file.

        Parameters
        ----------
        filename : str
            File name of the parquet archive to read.
        name_coords : list, optional
            Names of the coordinate variables in the ragged arrays

        Returns
        -------
        RaggedArray
            A ragged array instance
        """
        return cls.from_awkward(ak.from_parquet(filename), name_coords)

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, dim_traj: str = "traj", dim_obs: str = "obs"):
        """Populate a RaggedArray instance from an xarray Dataset instance.

        Parameters
        ----------
        ds : xr.Dataset
            Xarray Dataset from which to load the RaggedArray
        dim_traj : str, optional
            Name of the trajectories dimension in the xarray Dataset
        dim_obs : str, optional
            Name of the observations dimension in the xarray Dataset

        Returns
        -------
        RaggedArray
            A RaggedArray instance
        """
        coords = {}
        metadata = {}
        data = {}
        attrs_global = {}
        attrs_variables = {}

        attrs_global = ds.attrs

        for var in ds.coords.keys():
            coords[var] = ds[var].data
            attrs_variables[var] = ds[var].attrs

        for var in ds.data_vars.keys():
            if len(ds[var]) == ds.dims[dim_traj]:
                metadata[var] = ds[var].data
            elif len(ds[var]) == ds.dims[dim_obs]:
                data[var] = ds[var].data
            else:
                warnings.warn(
                    f"""
                    Variable '{var}' has unknown dimension size of 
                    {len(ds[var])}, which is not traj={ds.dims[dim_traj]} or 
                    obs={ds.dims[dim_obs]}; skipping.
                    """
                )
            attrs_variables[var] = ds[var].attrs

        return cls(coords, metadata, data, attrs_global, attrs_variables)

    @staticmethod
    def number_of_observations(
        rowsize_func: Callable[[int], int], indices: list, **kwargs
    ) -> np.array:
        """Iterate through the files and evaluate the number of observations.

        Parameters
        ----------
        rowsize_func : Callable[[int], int]]
            Function that returns the number observations of a trajectory from
            its identification number
        indices : list
            Identification numbers list to iterate

        Returns
        -------
        np.ndarray
            Number of observations of each trajectory
        """
        rowsize = np.zeros(len(indices), dtype="int")

        for i, index in tqdm(
            enumerate(indices),
            total=len(indices),
            desc="Retrieving the number of obs",
            ncols=80,
        ):
            rowsize[i] = rowsize_func(index, **kwargs)
        return rowsize

    @staticmethod
    def attributes(
        ds: xr.Dataset, name_coords: list, name_meta: list, name_data: list
    ) -> Tuple[dict, dict]:
        """Return global attributes and the attributes of all variables
        (name_coords, name_meta, and name_data) from an Xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            _description_
        name_coords : list
            Name of the coordinate variables to include in the archive
        name_meta : list, optional
            Name of metadata variables to include in the archive (default is [])
        name_data : list, optional
            Name of the data variables to include in the archive (default is [])

        Returns
        -------
        Tuple[dict, dict]
            The global and variables attributes
        """
        attrs_global = ds.attrs

        # coordinates, metadata, and data
        attrs_variables = {}
        for var in name_coords + name_meta + name_data:
            if var in ds.keys():
                attrs_variables[var] = ds[var].attrs
            else:
                warnings.warn(f"Variable {var} requested but not found; skipping.")

        return attrs_global, attrs_variables

    @staticmethod
    def allocate(
        preprocess_func: Callable[[int], xr.Dataset],
        indices: list,
        rowsize: list,
        name_coords: list,
        name_meta: list,
        name_data: list,
        **kwargs,
    ) -> Tuple[dict, dict, dict]:
        """
        Iterate through the files and fill for the ragged array associated
        with coordinates, and selected metadata and data variables.

        Parameters
        ----------
        preprocess_func : Callable[[int], xr.Dataset]
            Returns a processed xarray Dataset from an identification number.
        indices : list
            List of indices separating trajectory in the ragged arrays.
        rowsize : list
            List of the number of observations per trajectory.
        name_coords : list
            Name of the coordinate variables to include in the archive.
        name_meta : list, optional
            Name of metadata variables to include in the archive (Defaults to []).
        name_data : list, optional
            Name of the data variables to include in the archive (Defaults to []).

        Returns
        -------
        Tuple[dict, dict, dict]
            Dictionaries containing numerical data and attributes of coordinates, metadata and data variables.
        """

        # open one file to get dtype of variables
        ds = preprocess_func(indices[0], **kwargs)
        nb_traj = len(rowsize)
        nb_obs = np.sum(rowsize).astype("int")
        index_traj = rowsize_to_index(rowsize)

        # allocate memory
        coords = {}
        for var in name_coords:
            coords[var] = np.zeros(nb_obs, dtype=ds[var].dtype)

        metadata = {}
        for var in name_meta:
            try:
                metadata[var] = np.zeros(nb_traj, dtype=ds[var].dtype)
            except KeyError:
                warnings.warn(f"Variable {var} requested but not found; skipping.")

        data = {}
        for var in name_data:
            if var in ds.keys():
                data[var] = np.zeros(nb_obs, dtype=ds[var].dtype)
            else:
                warnings.warn(f"Variable {var} requested but not found; skipping.")
        ds.close()

        # loop and fill the ragged array
        for i, index in tqdm(
            enumerate(indices),
            total=len(indices),
            desc="Filling the Ragged Array",
            ncols=80,
        ):
            with preprocess_func(index, **kwargs) as ds:
                size = rowsize[i]
                oid = index_traj[i]

                for var in name_coords:
                    coords[var][oid : oid + size] = ds[var].data

                for var in name_meta:
                    try:
                        metadata[var][i] = ds[var][0].data
                    except KeyError:
                        warnings.warn(
                            f"Variable {var} requested but not found; skipping."
                        )

                for var in name_data:
                    if var in ds.keys():
                        data[var][oid : oid + size] = ds[var].data
                    else:
                        warnings.warn(
                            f"Variable {var} requested but not found; skipping."
                        )

        return coords, metadata, data

    def validate_attributes(self):
        """Validate that each variable has an assigned attribute tag."""
        for key in (
            list(self.coords.keys())
            + list(self.metadata.keys())
            + list(self.data.keys())
        ):
            if key not in self.attrs_variables:
                self.attrs_variables[key] = {}

    def to_xarray(self, cast_to_float32: bool = True):
        """Convert ragged array object to a xarray Dataset.

        Parameters
        ----------
        cast_to_float32 : bool, optional
            Cast all float64 variables to float32 (default is True). This option aims at
            minimizing the size of the xarray dataset.

        Returns
        -------
        xr.Dataset
            Xarray Dataset containing the ragged arrays and their attributes
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
        """Convert ragged array object to an Awkward Array.

        Returns
        -------
        ak.Array
            Awkward Array containing the ragged array and its attributes
        """
        index_traj = rowsize_to_index(self.metadata["rowsize"])
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
        """Export ragged array object to a NetCDF file.

        Parameters
        ----------
        filename : str
            Name of the NetCDF file to create.
        """

        self.to_xarray().to_netcdf(filename)

    def to_parquet(self, filename: str):
        """Export ragged array object to a parquet file.

        Parameters
        ----------
        filename : str
            Name of the parquet file to create.
        """
        ak.to_parquet(self.to_awkward(), filename)

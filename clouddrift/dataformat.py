import awkward as ak
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

        Args:
            array (ak.Array): Awkward Array instance to load the data from
            name_coords (list, optional): Names of the coordinate variables in the ragged arrays

        Returns:
            obj: RaggedArray instance
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
    ):
        """Generate ragged arrays archive from a list of trajectory files

        Args:
            indices (list): identification numbers list to iterate
            preprocess_func (Callable[[int], xr.Dataset]): returns a processed xarray Dataset from an identification number
            name_coords (list): Name of the coordinate variables to include in the archive
            name_meta (list, optional): Name of metadata variables to include in the archive (Defaults to [])
            name_data (list, optional): Name of the data variables to include in the archive (Defaults to [])
            rowsize_func (Optional[Callable[[int], int]], optional): returns the number of observations from an identification number (to speed up processing) (Defaults to None)

        Returns:
            obj: ragged array class object
        """
        # if no method is supplied, get the dimension from the preprocessing function
        rowsize_func = (
            rowsize_func if rowsize_func else lambda i: preprocess_func(i).dims["obs"]
        )
        rowsize = cls.number_of_observations(rowsize_func, indices)
        coords, metadata, data = cls.allocate(
            preprocess_func, indices, rowsize, name_coords, name_meta, name_data
        )
        attrs_global, attrs_variables = cls.attributes(
            preprocess_func(indices[0]), name_coords, name_meta, name_data
        )

        return cls(coords, metadata, data, attrs_global, attrs_variables)

    @classmethod
    def from_netcdf(cls, filename: str):
        """Read a ragged arrays archive from a NetCDF file.

        This is a thin wrapper around from_xarray().

        Args:
            filename (str): filename of NetCDF archive

        Returns:
            obj: ragged array class object
        """
        return cls.from_xarray(xr.open_dataset(filename))

    @classmethod
    def from_parquet(
        cls, filename: str, name_coords: Optional[list] = ["time", "lon", "lat", "ids"]
    ):
        """Read a ragged arrays archive from a parquet file

        Args:
            filename (str): filename of parquet archive
            name_coords (list, optional): Names of the coordinate variables in the ragged arrays

        Returns:
            obj: ragged array class object
        """
        return cls.from_awkward(ak.from_parquet(filename), name_coords)

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, dim_traj: str = "traj", dim_obs: str = "obs"):
        """Populate a RaggedArray instance from an xarray Dataset instance.

        Args:
            ds (xarray.Dataset): xarray Dataset from which to load the RaggedArray
            dim_traj (str, optional): Name of the trajectories dimension in the xarray Dataset
            dim_obs (str, optional): Name of the observations dimension in the xarray Dataset

        Returns:
            res (RaggedArray): A RaggedArray instance
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
        rowsize_func: Callable[[int], int], indices: list
    ) -> np.array:
        """Iterate through the files and evaluate the number of observations.

        Args:
            rowsize_func (Callable[[int], int]): returns number observations of a trajectory from its identification number
            indices (list): identification numbers list to iterate

        Returns:
            np.array: number of observations of each trajectory
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
        ds: xr.Dataset, name_coords: list, name_meta: list, name_data: list
    ) -> Tuple[dict, dict]:
        """Returns the global attributes and the attributes of all variables (name_coords, name_meta, and name_data) from a xr.Dataset

        Args:
            ds (xr.Dataset): _description_
            name_coords (list): Name of the coordinate variables to include in the archive
            name_meta (list): Name of metadata variables to include in the archive (Defaults to [])
            name_data (list): Name of the data variables to include in the archive (Defaults to [])

        Returns:
            Tuple[dict, dict]: the global and variables attributes
        """
        attrs_global = ds.attrs

        # coordinates, metadata, and data
        attrs_variables = {}
        for var in name_coords + name_meta + name_data:
            attrs_variables[var] = ds[var].attrs

        return attrs_global, attrs_variables

    @staticmethod
    def allocate(
        preprocess_func: Callable[[int], xr.Dataset],
        indices: list,
        rowsize: list,
        name_coords: list,
        name_meta: list,
        name_data: list,
    ) -> Tuple[dict, dict, dict]:
        """Iterate through the files and fill for the ragged array associated with coordinates, and selected metadata and data variables.

        Args:
            preprocess_func (Callable[[int], xr.Dataset]): returns a processed xarray Dataset from an identification number
            indices (list): list of indices separating trajectory in the ragged arrays
            rowsize (list): list of the number of observations per trajectory
            name_coords (list): Name of the coordinate variables to include in the archive
            name_meta (list, optional): Name of metadata variables to include in the archive (Defaults to [])
            name_data (list, optional): Name of the data variables to include in the archive (Defaults to [])

        Returns:
            Tuple[dict, dict, dict]: dictionaries containing numerical data and attributes of coordinates, metadata and data variables.
        """

        # open one file to get dtype of variables
        ds = preprocess_func(indices[0])
        nb_traj = len(rowsize)
        nb_obs = np.sum(rowsize).astype("int")
        index_traj = np.insert(np.cumsum(rowsize), 0, 0)

        # allocate memory
        coords = {}
        for var in name_coords:
            coords[var] = np.zeros(nb_obs, dtype=ds[var].dtype)

        metadata = {}
        for var in name_meta:
            metadata[var] = np.zeros(nb_traj, dtype=ds[var].dtype)

        data = {}
        for var in name_data:
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

                for var in name_coords:
                    coords[var][oid : oid + size] = ds[var].data

                for var in name_meta:
                    metadata[var][i] = ds[var][0].data

                for var in name_data:
                    data[var][oid : oid + size] = ds[var].data

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

    def to_xarray(self):
        """Convert ragged array object to a xarray Dataset.

        Returns:
            xr.Dataset: xarray Dataset containing the ragged arrays and their attributes
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

        Returns:
            ak.Array: Awkward Array containing the ragged arrays and their attributes
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
        """Export ragged array object to a NetCDF archive.

        Args:
            filename (str): filename of the NetCDF archive of ragged arrays
        """

        self.to_xarray().to_netcdf(filename)
        return

    def to_parquet(self, filename: str):
        """Export ragged array object to a parquet archive.

        Args:
            filename (str): filename of the parquet archive of ragged arrays
        """
        ak.to_parquet(self.to_awkward(), filename)
        return


def unpack_ragged(
    ragged_array: np.ndarray, rowsize: np.ndarray[int]
) -> list[np.ndarray]:
    """Unpack a ragged array into a list of regular arrays.

    Unpacking a ``np.ndarray`` ragged array is about 2 orders of magnitude
    faster than unpacking an ``xr.DataArray`` ragged array, so unless you need a
    ``DataArray`` as the result, we recommend passing ``np.ndarray`` as input.

    Parameters
    ----------
    ragged_array : array-like
        A ragged_array to unpack
    rowsize : array-like
        An array of integers whose values is the size of each row in the ragged
        array

    Returns
    -------
    list
        A list of array-likes with sizes that correspond to the values in
        rowsize, and types that correspond to the type of ragged_array

    Examples
    --------

    Unpacking longitude arrays from a ragged Xarray Dataset:

    .. code-block:: python

        lon = unpack_ragged(ds.lon, ds.rowsize) # return a list[xr.DataArray] (slower)
        lon = unpack_ragged(ds.lon.values, ds.rowsize) # return a list[np.ndarray] (faster)

    Looping over trajectories in a ragged Xarray Dataset to compute velocities
    for each:

    .. code-block:: python

        for lon, lat, time in list(zip(
            unpack_ragged(ds.lon.values, ds.rowsize),
            unpack_ragged(ds.lat.values, ds.rowsize),
            unpack_ragged(ds.time.values, ds.rowsize)
        )):
            u, v = velocity_from_position(lon, lat, time)
    """
    indices = np.insert(np.cumsum(np.array(rowsize)), 0, 0)
    return [ragged_array[indices[n] : indices[n + 1]] for n in range(indices.size - 1)]

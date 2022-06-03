from abc import abstractmethod
import typing
import json
import numpy as np
import xarray
import dask
from erddapy import ERDDAP


class data:
    """
    Abstract class to query the dataset
    """

    @abstractmethod
    def retrieve_drifter(self, id: int):
        raise NotImplementedError()

    @abstractmethod
    def retrieve_region(self, lon: list = None, lat: list = None, time: list = None):
        raise NotImplementedError()

    @abstractmethod
    def to_xarray(self) -> xarray.Dataset:
        raise NotImplementedError()


class erddap(data):
    """
    ERDDAP Description
    """

    def __init__(
        self,
        url="https://data.pmel.noaa.gov/generic/erddap/",
        protocol="tabledap",
        dataset_id="gdp_hourly_velocities",
    ):
        self.url = url
        self.protocol = protocol
        self.dataset_id = dataset_id
        self.response = "nc"
        self.server = ERDDAP(self.url, self.protocol, self.response)
        self.server.dataset_id = dataset_id

    def retrieve_drifter(self, id: int):
        """

        :param id:
        :return:
        """
        self.server.constraints = {"ID=": str(id)}

    def retrieve_region(self, lon: list = None, lat: list = None, time: list = None):
        """

        :param lon:
        :param lat:
        :param time:
        :return:
        """
        constraints = {}

        if lon:
            constraints["longitude>="] = lon[0]
            constraints["longitude<="] = lon[1]

        if lat:
            constraints["latitude>="] = lat[0]
            constraints["latitude<="] = lat[1]

        if time:
            constraints["time>="] = time[0]
            constraints["time<="] = time[1]

        self.server.constraints = constraints

    def print_constraints(self):
        """

        :return:
        """
        print(f"All variables in this dataset:\n{self.server.variables}")
        print(
            f"\nConstraints of this dataset:\n{json.dumps(self.server.constraints, indent=1)}"
        )

    def to_xarray(self) -> xarray.Dataset:
        return self.server.to_xarray()


class local(data):
    """
    Local Description
    """

    def __init__(self, path_obs: str, path_traj: str):
        self.path_traj = path_traj
        self.path_obs = path_obs
        self.ds_traj = xarray.open_dataset(self.path_traj)
        self.ds_obs = xarray.open_dataset(
            self.path_obs,
            # one trajectory per chunk
            chunks={"obs": tuple(self.ds_traj["rowsize"].data.tolist())},
        )
        self.number_traj = self.ds_traj.dims["traj"]
        self.number_obs = self.ds_obs.dims["obs"]
        self.traj_idx = np.insert(np.cumsum(self.ds_traj["rowsize"].values), 0, 0)

    def retrieve_drifter(self, id: int):
        """
        :param id:
        :return:
        """
        if isinstance(id, int):
            if np.isin(id, self.ds_traj.ID):
                j = np.where(self.ds_traj["ID"].data == id)[0][0]
                ds_subset = self.ds_obs.isel(
                    obs=slice(self.traj_idx[j], self.traj_idx[j + 1])
                )
            else:
                print("Error: Invalid id number.\n")
                return None

        elif isinstance(id, typing.Iterable):
            mask = np.empty(0, dtype="int")

            for i in id:
                if not np.isin(i, self.ds_traj.ID):
                    print("Error: Invalid id number %d.\n" % i)
                    return None

            j = []
            for i in id:
                j.append(np.where(self.ds_traj["ID"].data == i)[0][0])
                mask = np.hstack(
                    (mask, np.arange(self.traj_idx[j[-1]], self.traj_idx[j[-1] + 1]))
                )
            ds_subset = self.ds_obs.isel(obs=mask)

        return ds_subset.compute()

    def retrieve_region(self, lon: list = None, lat: list = None, time: list = None):
        """

        :param lon:
        :param lat:
        :param time:
        :return:
        """
        mask = np.ones(self.number_obs, dtype="bool")

        if lon:  # TODO: deal with ranges across dateline
            mask &= (self.ds_obs.coords["longitude"] >= lon[0]).values
            mask &= (self.ds_obs.coords["longitude"] <= lon[1]).values

        if lat:
            mask &= (self.ds_obs.coords["latitude"] >= lat[0]).values
            mask &= (self.ds_obs.coords["latitude"] <= lat[1]).values

        if time:
            mask &= (self.ds_obs.coords["time"] >= np.datetime64(time[0])).values
            mask &= (self.ds_obs.coords["time"] <= np.datetime64(time[1])).values

        ds_subset = self.ds_obs.isel(obs=np.where(mask)[0])

        return ds_subset.compute()

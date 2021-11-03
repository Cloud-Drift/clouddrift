from abc import abstractmethod
import numpy as np
import xarray
import dask
import typing
import json
from erddapy import ERDDAP


class data:
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
    def __init__(self, url="https://data.pmel.noaa.gov/generic/erddap/",
                 protocol="tabledap", dataset_id="gdp_hourly_velocities"):
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
        print(f"\nConstraints of this dataset:\n{json.dumps(self.server.constraints, indent=1)}")

    def to_xarray(self) -> xarray.Dataset:
        return self.server.to_xarray()


class local(data):
    def __init__(self, path: str):
        self.path = path
        self.ds = xarray.open_dataset(path, chunks={"obs": 10000000})
        #self.ds = xarray.open_zarr(path)
        self.number_traj = self.ds.dims['traj']
        self.number_obs = self.ds.dims['obs']
        self.traj_idx = np.insert(np.cumsum(self.ds["count"].values), 0, 0)

    def retrieve_drifter(self, id: int):
        """
        :param id:
        :return:
        """
        if isinstance(id, int):
            j = int(np.where(self.ds['id'] == id)[0])
            ds_subset = self.ds.isel(traj=[j], obs=slice(self.traj_idx[j], self.traj_idx[j + 1]))
        
        elif isinstance(id, typing.Iterable):
            mask = np.empty(0, dtype='int')
            j = []
            for i in id:
                j.append(int(np.where(self.ds['id'] == i)[0]))
                mask = np.hstack((mask, np.arange(self.traj_idx[j[-1]], self.traj_idx[j[-1] + 1])))
            ds_subset = self.ds.isel(traj=j, obs=mask)
            
        return ds_subset.compute()

    def retrieve_region(self, lon: list = None, lat: list = None, time: list = None):
        """

        :param lon:
        :param lat:
        :param time:
        :return:
        """
        mask = np.ones(self.number_obs, dtype='bool')

        if lon:  # TODO: deal with ranges across dateline
            mask &= (self.ds.coords["longitude"] >= lon[0]).values
            mask &= (self.ds.coords["longitude"] <= lon[1]).values

        if lat:
            mask &= (self.ds.coords["latitude"] >= lat[0]).values
            mask &= (self.ds.coords["latitude"] <= lat[1]).values

        if time:
            mask &= (self.ds.coords["time"] >= np.datetime64(time[0])).values
            mask &= (self.ds.coords["time"] <= np.datetime64(time[1])).values

        idx, count = np.unique(self.ds["ids"][mask], return_counts=True)
        ds_subset = self.ds.isel(traj=idx, obs=np.where(mask)[0])
        ds_subset['count'].data = count

        return ds_subset.compute()

import awkward._v2 as ak
import xarray as xr
import numpy as np
from collections.abc import Callable
from typing import Tuple
import concurrent.futures
from tqdm import tqdm


class create_ragged_array:
    def __init__(self, indices: list, vars_coords: dict, vars_meta: list, vars_data: list, 
                 preprocess_func: Callable[[int], xr.Dataset], rowsize_func: Callable[[int], int] = None):
        self.indices = indices
        self.preprocess_func = preprocess_func
        self.rowsize_func = rowsize_func if rowsize_func else lambda i: self.preprocess_func(i).dims['obs']
        self.rowsize = self.number_of_observations()
        self.index_traj = np.insert(np.cumsum(self.rowsize), 0, 0)
        self.nb_traj = len(self.rowsize)
        self.nb_obs = np.sum(self.rowsize).astype('int')
        self.variables_coords = vars_coords
        self.variables_metadata = vars_meta
        self.variables_data = vars_data
        self.attrs_global, self.attrs_variables = self.attributes()
        self.coords, self.metadata, self.data = self.allocate_data()
        self.fill_ragged_arrays()

    def number_of_observations(self) -> np.array:
        '''
        Load files and get the size of the observations.
        '''
        with concurrent.futures.ThreadPoolExecutor() as exector:
            rowsize = list(tqdm(exector.map(self.rowsize_func, self.indices), total=len(self.indices), desc='Calculating the number of observations'))

        return np.array(rowsize, dtype='int')

    def attributes(self) -> Tuple[dict,dict]:
        ds = self.preprocess_func(self.indices[0])  # open file to get atts

        attrs_global = ds.attrs
        attrs_variables = {}

        # coordinates
        for var in self.variables_coords.keys():
            attrs_variables[var] = ds[self.variables_coords[var]].attrs

        # metadata
        for var in self.variables_metadata:
            attrs_variables[var] = ds[var].attrs

        # observations
        for var in self.variables_data:
            attrs_variables[var] = ds[var].attrs
 
        ds.close()

        return attrs_global, attrs_variables

    def allocate_data(self) -> Tuple[dict, dict, dict]:
        '''
        Reserve the space for the ragged array associated with all variables
        '''
        # open one file to get dtype of variables
        ds = self.preprocess_func(self.indices[0])  
        
        coords = {}
        for var in self.variables_coords.keys():
            coords[var] = np.zeros(self.nb_obs, dtype=ds[self.variables_coords[var]].dtype)

        metadata = {}
        for var in self.variables_metadata:
            metadata[var] = np.zeros(self.nb_traj, dtype=ds[var].dtype)

        data = {}
        for var in self.variables_data:
            data[var] = np.zeros(self.nb_obs, dtype=ds[var].dtype)

        ds.close()
        
        return coords, metadata, data
    
    def fill_ragged_arrays(self):
        '''
        Fill the ragged array datastructure by iterating through the identification numbers
        '''
        
        for i, index in tqdm(enumerate(self.indices), total=len(self.indices), desc='Filling the ragged array'):
            ds = self.preprocess_func(index)
            
            size = ds.dims['obs']
            oid = self.index_traj[i]
            
            for var in self.variables_coords.keys():
                self.coords[var][oid:oid+size] = ds[self.variables_coords[var]].data

            for var in self.variables_metadata:
                self.metadata[var][i] = ds[var][0].data

            for var in self.variables_data:
                self.data[var][oid:oid+size] = ds[var].data
        
            ds.close()
            
        return
    
    def to_netcdf(self, filename: str):
        '''
        Export ragged array dataset to NetCDF archive
        
        Args: filename: path of the archive
        '''
        self.to_xarray().to_netcdf(filename)
        return
    
    def to_parquet(self, filename: str):
        '''
        Export ragged array dataset to parquet archive
        
        Args: filename: path of the archive
        '''
        ak.to_parquet(self.to_awkward(), filename)
        return
    
    def to_xarray(self):
        '''
        Output the ragged array dataformat to an xr.Dataset
        '''
        xr_coords = {}
        for var in self.coords.keys():
            xr_coords[var] = (['obs'], self.coords[var], self.attrs_variables[var])
        
        xr_data = {}
        for var in self.metadata.keys():
            xr_data[var] = (['traj'], self.metadata[var], self.attrs_variables[var])
        
        for var in self.data.keys():
            xr_data[var] = (['obs'], self.data[var], self.attrs_variables[var])
                
        return xr.Dataset(
            coords = xr_coords,
            data_vars = xr_data,
            attrs = self.attrs_global
        )
    
    def to_awkward(self):
        '''
        Output the ragged array dataformat to an Awkward Array archive
        '''
        offset = ak.index.Index64(self.index_traj)
        
        data = []
        for var in self.coords.keys():
            data.append(
                ak.contents.ListOffsetArray(offset, ak.contents.NumpyArray(self.coords[var]), parameters={'attrs': self.attrs_variables[var]})
            )
        for var in self.data.keys():
            data.append(
                ak.contents.ListOffsetArray(offset, ak.contents.NumpyArray(self.data[var]), parameters={'attrs': self.attrs_variables[var]})
            )
        data_names = list(self.variables_coords.keys()) +  self.variables_data
        
        metadata = []
        for var in self.metadata.keys():
            metadata.append(
                ak.with_parameter(self.metadata[var], 'attrs', self.attrs_variables[var], highlevel=False)
            )
        metadata_names = self.variables_metadata.copy()
        
        # include the data inside the metadata list as a nested array
        metadata_names.append('obs')
        metadata.append(ak.Array(ak.contents.RecordArray(data, data_names)).layout)
        
        return ak.Array(
            ak.contents.RecordArray(metadata, metadata_names, parameters={'attrs': self.attrs_global})
        )


def read_from_netcdf(filename: str):
    '''
    param: filename: path of the archive
    return: Awkward Array from a NetCDF archive
    '''
    ds = xr.open_dataset(filename)
    index_traj = np.insert(np.cumsum(ds.rowsize), 0, 0)
    offset = ak.index.Index64(index_traj)
    nb_traj = ds.dims['traj']
    nb_obs = ds.dims['obs']
    
    metadata = []
    metadata_names = []
    data = []
    data_names = []
    
    for var in ds.coords.keys():
        data.append(
            ak.contents.ListOffsetArray(offset, ak.contents.NumpyArray(ds[var]), parameters={'attrs': ds[var].attrs})
        )
        data_names.append(var)
        
    for var in ds.data_vars.keys():
        if len(ds[var]) == nb_traj:
            metadata.append(ak.with_parameter(ds[var].data, 'attrs', ds[var].attrs, highlevel=False))
            metadata_names.append(var)
        elif len(ds[var]) == nb_obs:
            data.append(
                ak.contents.ListOffsetArray(offset, ak.contents.NumpyArray(ds[var]), parameters={'attrs': ds[var].attrs})
            )
            data_names.append(var)
        else:
            print(f"Error: variable '{var}' has unknown dimension size of {len(ds[var])}, which is not traj={nb_traj} or obs={nb_obs}.")
    ds.close()
    
    # include the data inside the metadata list as a nested array
    metadata_names.append('obs')
    metadata.append(ak.Array(ak.contents.RecordArray(data, data_names)).layout)
    
    return ak.Array(ak.contents.RecordArray(metadata, metadata_names, parameters={'attrs': ds.attrs}))


def read_from_parquet(filename: str):
    '''
    param: filename: path of the archive
    return: Awkward Array from a parquet archive
    '''
    return ak.from_parquet(filename)  # lazy=True not available yet 
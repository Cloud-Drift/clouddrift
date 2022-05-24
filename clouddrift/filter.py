import awkward._v2 as ak
import numpy as np
import functools


def region(ds: ak.Array, lon: list = None, lat: list = None, days: list = None) -> ak.Array:
    '''Subset the dataset for a region in space and time
    
    Args:
        ds: Awkward Array
        lon: longitude slice of the subregion
        lat: latitude slice of the subregion
        days: days slice of the subregion
    
    Returns: 
        ds_subset: Dataset of the subregion
    '''
    
    if lon is None:
        lon = [ak.min(ds.obs.lon), ak.max(ds.obs.lon)]
    
    if lat is None:
        lat = [ak.min(ds.obs.lat), ak.max(ds.obs.lat)]
        
    if days is None:
        days = [ak.min(ds.obs.time), ak.max(ds.obs.time)]
    
    mask = functools.reduce(np.logical_and, 
                            (
                                ds.obs.lon >= lon[0], 
                                ds.obs.lon <= lon[1],
                                ds.obs.lat >= lat[0],
                                ds.obs.lat <= lat[1],
                                ds.obs.time >= days[0],
                                ds.obs.time <= days[1],
                            )
                           )
    
    ds_s = ak.copy(ds)
    mask_id = np.in1d(ds_s.ID, np.unique(ak.flatten(ds_s.obs.ids[mask])))
    ds_s = ds_s[mask_id]  # mask for variables with dimension ['traj']
    ds_s.obs = ds_s.obs[mask[mask_id]] # mask for variables with dimension ['obs']
    
    return copy(ds_s)
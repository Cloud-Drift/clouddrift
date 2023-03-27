import xarray as xr
import numpy as np
import warnings
from typing import Union


def mask_var(var: xr.DataArray, criterion: Union[tuple, list, bool, float, int]) -> xr.DataArray:
    """Return the mask of a subset of the data matching a test criterion.

    Parameters
    ----------
    var : xr.DataArray
        DataArray to be subset by the criterion
    criterion : Union[tuple, list, bool, float, int]
        The criterion can take three forms:
        - tuple: (min, max) defining a range
        - list: [value1, value2, valueN] defining multiples values
        - scalar: value defining a single value

    Examples
    --------
    >>> x = xr.DataArray(data=np.arange(0, 5))
    >>> mask_var(x, (2, 4))
    <xarray.DataArray (dim_0: 5)> 
    array([False, False,  True,  True,  True])
    Dimensions without coordinates: dim_0
    
    >>> mask_var(x, [0, 2, 4])
    <xarray.DataArray (dim_0: 5)>
    array([ True, False, True,  False, True])
    Dimensions without coordinates: dim_0

    >>> mask_var(x, 4)
    <xarray.DataArray (dim_0: 5)>
    array([False, False, False,  True, False])
    Dimensions without coordinates: dim_0

    Returns
    -------
    mask : xr.DataArray
        The mask of the subset of the data matching the criteria
    """
    if isinstance(criterion, tuple):  # min/max defining range
        mask = np.logical_and(var >= criterion[0], var <= criterion[1])
    elif isinstance(criterion, list):  # select multiple values
        mask = xr.zeros_like(var)
        for v in criterion:
            mask = np.logical_or(mask, var == v)
    else:  # select one specific value
        mask = var == criterion
    return mask


def subset(ds: xr.Dataset, criteria: dict) -> xr.Dataset:
    """Subset the dataset as a function of one or many criteria. The criteria are passed as a dictionary, where
    a variable to subset is assigned to either a range (valuemin, valuemax), a list [value1, value2, valueN],
    or a single value.

    Parameters
    ----------
    ds : xr.Dataset
        Lagrangian dataset stored in two-dimensional or ragged array format
    criteria : dict
        dictionary containing the variables and the ranges/values to subset

    Returns
    -------
    xr.Dataset
        subset Dataset matching the criterion(a)

    Examples
    --------
    Criteria are combined on any data or metadata variables part of the Dataset.

    To subset between a range of values:
    >>> subset(ds, {"lon": (min_lon, max_lon), "lat": (min_lat, max_lat)})
    >>> subset(ds, {"time": (min_time, max_time)})

    To select multiples values:
    >>> subset(ds, {"ID": [1, 2, 3]})

    To select a specific value:
    >>> subset(ds, {"drogue_status": True})

    Raises
    ------
    ValueError
        If one of the variable in a criterion is not found in the Dataset
    """    
    mask_traj = xr.DataArray(data=np.ones(ds.dims["traj"], dtype="bool"), dims=["traj"])
    mask_obs = xr.DataArray(data=np.ones(ds.dims["obs"], dtype="bool"), dims=["obs"])

    for key in criteria.keys():
        if key in ds:
            if ds[key].dims == ("traj",):
                mask_traj = np.logical_and(mask_traj, mask_var(ds[key], criteria[key]))
            elif ds[key].dims == ("obs",):
                mask_obs = np.logical_and(mask_obs, mask_var(ds[key], criteria[key]))
        else:
            raise ValueError(f"Unknown variable '{key}'.")

    # remove data when trajectories are filtered
    traj_idx = np.insert(np.cumsum(ds["rowsize"].values), 0, 0)
    for i in np.where(~mask_traj)[0]:
        mask_obs[slice(traj_idx[i], traj_idx[i + 1])] = False

    # remove trajectory completely filtered in mask_obs
    mask_traj = np.logical_and(
        mask_traj, np.in1d(ds["ID"], np.unique(ds["ids"].isel({"obs": mask_obs})))
    )

    if not any(mask_traj):
        warnings.warn("No data matches the criteria; returning an empty dataset.")
        return xr.Dataset()
    else:
        # update rowsize
        id_count = np.bincount(ds.ids[mask_obs])
        ds["rowsize"].values[mask_traj] = [id_count[i] for i in ds.ID[mask_traj]]
        # apply the filtering for both dimensions
        return ds.isel({"traj": mask_traj, "obs": mask_obs})

import awkward as ak
import numpy as np


def mask_var(var: xr.DataArray, value):
    """Return the mask of a subset of the data matching a test criterion.

    Args:
        var (xr.DataArray): ak.Array
        value: tuple, list or scalar defining a test criterion

    Returns:
        mask (xr.DataArray): where values corresponding to the criterion are True
    """

    if isinstance(value, tuple):  # min/max defining range
        mask = np.logical_and(var >= min(value), var <= max(value))
    elif isinstance(value, list):  # select multiples
        mask = ak.zeros_like(var)
        for v in value:
            mask = np.logical_or(mask, var == v)
    else:  # select one
        mask = var == value
    return mask


def subset(ds: xr.Dataset, criteria: dict) -> xr.Dataset:
    """Subset the dataset as a function of one or many criteria. The criteria are passed as a dictionary, where
    a variable to subset is assigned to either a range (valuemin, valuemax), a list [value1, value2, valueN],
    or a single value.

    Args:
        ds (xr.Dataset): dataset
        criteria (dict): dictionary containing the variables and the ranges/values to retrieve

    Returns:
        ds_subset: subset xr.Dataset

    Usage:
        Operation can be combined, and any data or metadata variables part of the Dataset can
        be used as a criterion. Criterion can be defined using three datatypes:

        Tuple to subset between a range of values:
        >>> subset(ds, {"lon": (min_lon, max_lon), "lat": (min_lat, max_lat)})  # extract a region
        >>> subset(ds, {"time": (min_time, max_time)})  # extract a temporal range

        A list to select multiples values.
        >>> subset(ds, {"ID": [1, 2, 3]})  # different IDs

        A scalar for selecting a specific value.
        >>> subset(ds, {"drogue_status": True})  # extract segment of trajectory with drogue
    """
    mask_traj = xr.DataArray(data=np.ones(ds.dims["traj"]), dims=["traj"])
    mask_obs = xr.DataArray(data=np.ones(ds.dims["obs"]), dims=["obs"])

    for key in criteria.keys():
        if key in ds:
            if ds[key].dims == ("traj",):
                mask_traj = np.logical_and(mask_traj, mask_var(ds[key], criteria[key]))
            elif ds[key].dims == ("obs",):
                mask_obs = np.logical_and(mask_obs, mask_var(ds[key], criteria[key]))
        else:
            raise ValueError(f"Unknown variable '{key}'.")

    # remove trajectory completely filtered in mask_obs
    mask_traj = np.logical_and(
        mask_traj, np.in1d(ds.ID, np.unique(ds.ids.sel({"obs": b})))
    )

    if not any(mask_traj):
        warnings.warn("Empty set.")
        return xr.Dataset()
    else:  # apply the filtering for both dimensions
        return ds.isel({"traj": mask_traj, "obs": mask_obs})

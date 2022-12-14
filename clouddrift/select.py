import awkward as ak
import numpy as np


def mask_var(var: ak.Array, value):
    """Return the mask of a subset of the data matching a test criterion.

    Args:
        var (ak.Array): ak.Array
        value: tuple, list or scalar defining a test criterion

    Returns:
        mask (Ak.Array): where values corresponding to the criterion are True
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


def subset(ds: ak.Array, criteria: dict) -> ak.Array:
    """Subset the dataset as a function of one or many criteria. The criteria are passed as a dictionary, where
    a variable to subset is assigned to either a range (valuemin, valuemax), a list [value1, value2, valueN],
    or a single value.

    Args:
        ds (ak.Array): dataset
        criteria (dict): dictionary containing the variables and the ranges/values to retrieve

    Returns:
        ds_subset: subset ak.Array Dataset

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

    mask_traj = ak.ones_like(ds[ds.fields[0]], dtype="bool")
    mask_obs = ak.ones_like(ds.obs[ds.obs.fields[0]], dtype="bool")

    for key in criteria.keys():
        if key in ds.fields:
            mask_traj = np.logical_and(mask_traj, mask_var(ds[key], criteria[key]))
        elif key in ds.obs.fields:
            mask_obs = np.logical_and(mask_obs, mask_var(ds.obs[key], criteria[key]))
        else:
            raise ValueError(f"Unknown variable '{key}'.")

    # mask id not in mask_obs
    mask_traj = np.logical_and(mask_traj, ak.any(mask_obs, 1))

    if not ak.any(mask_traj):
        print("Empty set.")
        return ak.Array([])
    else:  # apply the filtering for both dimensions
        ds_subset = ak.to_packed(ds[mask_traj])
        ds_subset = ak.with_field(
            ds_subset, ak.to_packed(ds.obs[mask_obs][mask_traj]), "obs"
        )
        ds_subset = ak.with_field(
            ds_subset, ak.Array([len(x) for x in ds_subset.obs.ids]), "rowsize"
        )
        return ds_subset

"""
Functions to analyze pairs of contiguous data segments.
"""
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union

array_like = Union[
    list[float], np.ndarray[float], pd.Series, xr.DataArray
]


def pair_time_overlap(
    time1: array_like,
    time2: array_like,
    tolerance,
) -> np.ndarray[bool]:
    """Given two arrays of times (or any other monotonically increasing
    quantity), return boolean masks for the overlapping times.

    Parameters
    ----------
    time1 : array_like
        First array of times.
    time2 : array_like
        Second array of times.
    tolerance : float
        Tolerance for the overlap. If the overlap is within this tolerance,
        the times are considered to overlap.

    Returns
    -------
    overlap1 : np.ndarray[bool]
        Boolean mask for the overlapping times in `time1`.
    overlap2 : np.ndarray[bool]
        Boolean mask for the overlapping times in `time2`.

    Examples
    --------
    >>> time1 = np.arange(4)
    >>> time2 = np.arange(2, 6)
    >>> pair_time_overlap(time1, time2, 0.5)
    (array([False,  False,  True, True]), array([ True,  True, False, False]))
    >>> pair_time_overlap(time1, time2, 1.5)
    (array([False,  True,  True, True]), array([ True,  True,  True, False]))
    """
    time1_min, time1_max = np.min(time1), np.max(time1)
    time2_min, time2_max = np.min(time2), np.max(time2)
    overlap_start = max(time1_min, time2_min) - tolerance
    overlap_end = min(time1_max, time2_max) + tolerance
    overlap1 = (time1 >= overlap_start) & (time1 <= overlap_end)
    overlap2 = (time2 >= overlap_start) & (time2 <= overlap_end)
    return overlap1, overlap2

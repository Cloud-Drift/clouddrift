"""
Functions to analyze pairs of contiguous data segments.
"""
from clouddrift import sphere
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Tuple, Union

array_like = Union[list[float], np.ndarray[float], pd.Series, xr.DataArray]


def chance_pair(
    lon1: array_like,
    lat1: array_like,
    lon2: array_like,
    lat2: array_like,
    time1: Optional[array_like] = None,
    time2: Optional[array_like] = None,
    space_tolerance: Optional[float] = 0,
    time_tolerance: Optional[float] = 0,
):
    """Given two arrays of longitudes and latitudes, return the chance of
    finding a pair of points that overlap in space and time.

    Parameters
    ----------
    lon1 : array_like
        First array of longitudes in degrees.
    lat1 : array_like
        First array of latitudes in degrees.
    lon2 : array_like
        Second array of longitudes in degrees.
    lat2 : array_like
        Second array of latitudes in degrees.
    time1 : array_like, optional
        First array of times.
    time2 : array_like, optional
        Second array of times.
    space_tolerance : float, optional
        Tolerance in meters for the spatial overlap. If the overlap is within
        this tolerance, the pair is considered to be a chance pair. Default is
        0, or no tolerance.
    time_tolerance : float, optional
        Tolerance for the temporal overlap. If the overlap is within this
        tolerance, and the space_tolerance condition is satisfied, the pair is
        considered a chance pair.. Default is 0, or no tolerance.

    Returns
    -------
    indices1 : np.ndarray[int]
        Indices of the first array that satisfy chance pair criteria.
    indices2 : np.ndarray[int]
        Indices of the second array that satisfy chance pair criteria.

    Examples
    --------
    TODO
    """
    time_present = time1 is not None and time2 is not None

    # If time is provided, subset the trajectories to the overlapping times.
    if time_present:
        overlap1, overlap2 = pair_time_overlap(time1, time2, time_tolerance)
        lon1 = lon1[overlap1]
        lat1 = lat1[overlap1]
        time1 = time1[overlap1]
        lon2 = lon2[overlap2]
        lat2 = lat2[overlap2]
        time2 = time2[overlap2]

    space_tolerance_degrees = np.degrees(space_tolerance / sphere.EARTH_RADIUS_METERS)

    bbox_overlap1, bbox_overlap2 = pair_bounding_box_overlap(
        lon1, lat1, lon2, lat2, space_tolerance_degrees
    )

    lon1 = lon1[bbox_overlap1]
    lat1 = lat1[bbox_overlap1]
    lon2 = lon2[bbox_overlap2]
    lat2 = lat2[bbox_overlap2]
    time1 = time1[bbox_overlap1] if time_present else None
    time2 = time2[bbox_overlap2] if time_present else None

    space_distance = pair_space_distance(lon1, lat1, lon2, lat2)
    chance_mask = space_distance <= space_tolerance

    if time_present:
        time_distance = pair_time_distance(time1, time2)
        chance_mask_time = time_distance <= time_tolerance
        chance_mask = chance_mask & chance_mask_time

    indices2, indices1 = np.where(chance_mask)

    # FIXME: We return indices for subsetted arrays, not the original arrays.
    # FIXME: We should return the indices for the original arrays.

    return indices1, indices2


def pair_bounding_box_overlap(
    lon1: array_like,
    lat1: array_like,
    lon2: array_like,
    lat2: array_like,
    tolerance: Optional[float] = 0,
) -> Tuple[np.ndarray[bool], np.ndarray[bool]]:
    """Given two arrays of longitudes and latitudes, return boolean masks for
    their overlapping bounding boxes.

    Parameters
    ----------
    lon1 : array_like
        First array of longitudes in degrees.
    lat1 : array_like
        First array of latitudes in degrees.
    lon2 : array_like
        Second array of longitudes in degrees.
    lat2 : array_like
        Second array of latitudes in degrees.
    tolerance : float, optional
        Tolerance in degrees for the overlap. If the overlap is within this
        tolerance, the bounding boxes are considered to overlap. Default is 0,
        or no tolerance.

    Returns
    -------
    overlap1 : np.ndarray[bool]
        Boolean mask for ``lon1`` and ``lat1`` that indicates where their bounding
        box overlaps with that of ``lon2`` and ``lat2``.
    overlap2 : np.ndarray[bool]
        Boolean mask for ``lon2`` and ``lat2`` that indicates where their bounding
        box overlaps with that of ``lon1`` and ``lat1``.

    Examples
    --------
    >>> lon1 = [0, 0, 1, 1]
    >>> lat1 = [0, 0, 1, 1]
    >>> lon2 = [1, 1, 2, 2]
    >>> lat2 = [1, 1, 2, 2]
    >>> pair_bounding_box_overlap(lon1, lat1, lon2, lat2, 0.5)
    (array([False, False,  True,  True]), array([ True,  True, False, False]))
    """
    # First get the bounding box of each trajectory.
    lon1_min, lon1_max = np.min(lon1), np.max(lon1)
    lat1_min, lat1_max = np.min(lat1), np.max(lat1)
    lon2_min, lon2_max = np.min(lon2), np.max(lon2)
    lat2_min, lat2_max = np.min(lat2), np.max(lat2)

    # TODO handle trajectories that cross the dateline

    bounding_boxes_overlap = (
        (lon1_min <= lon2_max + tolerance)
        & (lon1_max >= lon2_min - tolerance)
        & (lat1_min <= lat2_max + tolerance)
        & (lat1_max >= lat2_min - tolerance)
    )

    # Now check if the trajectories overlap within the bounding box.
    if bounding_boxes_overlap:
        overlap_start = (
            max(lon1_min, lon2_min) - tolerance,  # West
            max(lat1_min, lat2_min) - tolerance,  # South
        )
        overlap_end = (
            min(lon1_max, lon2_max) + tolerance,  # East
            min(lat1_max, lat2_max) + tolerance,  # North
        )
        overlap1 = (
            (lon1 >= overlap_start[0])
            & (lon1 <= overlap_end[0])
            & (lat1 >= overlap_start[1])
            & (lat1 <= overlap_end[1])
        )
        overlap2 = (
            (lon2 >= overlap_start[0])
            & (lon2 <= overlap_end[0])
            & (lat2 >= overlap_start[1])
            & (lat2 <= overlap_end[1])
        )
        return overlap1, overlap2
    else:
        return np.zeros_like(lon1, dtype=bool), np.zeros_like(lon2, dtype=bool)


def pair_space_distance(
    lon1: array_like,
    lat1: array_like,
    lon2: array_like,
    lat2: array_like,
) -> np.ndarray[float]:
    """Given two arrays of longitudes and latitudes, return the distance
    on a sphere between all pairs of points.

    Parameters
    ----------
    lon1 : array_like
        First array of longitudes in degrees.
    lat1 : array_like
        First array of latitudes in degrees.
    lon2 : array_like
        Second array of longitudes in degrees.
    lat2 : array_like
        Second array of latitudes in degrees.

    Returns
    -------
    distance : np.ndarray[float]
        Array of distances between all pairs of points.

    Examples
    --------
    >>> lon1 = [0, 0, 1, 1]
    >>> lat1 = [0, 0, 1, 1]
    >>> lon2 = [1, 1, 2, 2]
    >>> lat2 = [1, 1, 2, 2]
    >>> pair_space_distance(lon1, lat1, lon2, lat2)
    array([[157424.62387233, 157424.62387233,      0.        ,
             0.        ],
       [157424.62387233, 157424.62387233,      0.        ,
             0.        ],
       [314825.26360286, 314825.26360286, 157400.64794884,
        157400.64794884],
       [314825.26360286, 314825.26360286, 157400.64794884,
        157400.64794884]])
    """
    # Create longitude and latitude matrices from arrays to compute distance
    lon1_2d, lon2_2d = np.meshgrid(lon1, lon2, copy=False)
    lat1_2d, lat2_2d = np.meshgrid(lat1, lat2, copy=False)

    # Compute distance between all pairs of points
    distance = sphere.distance(lon1_2d, lat1_2d, lon2_2d, lat2_2d)

    return distance


def pair_time_distance(
    time1: array_like,
    time2: array_like,
) -> np.ndarray[float]:
    """Given two arrays of times (or any other monotonically increasing
    quantity), return the distance between all pairs of times.

    Parameters
    ----------
    time1 : array_like
        First array of times.
    time2 : array_like
        Second array of times.

    Returns
    -------
    distance : np.ndarray[float]
        Array of distances between all pairs of times.

    Examples
    --------
    >>> time1 = np.arange(4)
    >>> time2 = np.arange(2, 6)
    >>> pair_time_distance(time1, time2)
    array([[2, 1, 0, 1],
           [3, 2, 1, 0],
           [4, 3, 2, 1],
           [5, 4, 3, 2]])
    """
    # Create time matrices from arrays to compute distance
    time1_2d, time2_2d = np.meshgrid(time1, time2, copy=False)

    # Compute distance between all pairs of times
    distance = np.abs(time1_2d - time2_2d)

    return distance


def pair_time_overlap(
    time1: array_like,
    time2: array_like,
    tolerance: Optional[float] = 0,
) -> Tuple[np.ndarray[bool], np.ndarray[bool]]:
    """Given two arrays of times (or any other monotonically increasing
    quantity), return boolean masks for the overlapping times.

    Although higher-level array containers like xarray and pandas are supported
    for input arrays, this function is an order of magnitude faster when passing
    in numpy arrays.

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

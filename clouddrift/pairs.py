"""
Functions to analyze pairs of contiguous data segments.
"""

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from clouddrift import ragged, sphere

array_like = Union[list[float], np.ndarray[float], pd.Series, xr.DataArray]


def chance_pair(
    lon1: array_like,
    lat1: array_like,
    lon2: array_like,
    lat2: array_like,
    time1: Optional[array_like] = None,
    time2: Optional[array_like] = None,
    space_distance: Optional[float] = 0,
    time_distance: Optional[float] = 0,
):
    """Given two sets of longitudes, latitudes, and times arrays, return in pairs
    the indices of collocated data points that are within prescribed distances
    in space and time. Also known as chance pairs.

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
    space_distance : float, optional
        Maximum allowable space distance in meters for a pair to qualify as chance pair.
        If the separation is within this distance, the pair is considered to be
        a chance pair. Default is 0, or no distance, i.e. the positions must be
        exactly the same.
    time_distance : float, optional
        Maximum allowable time distance for a pair to qualify as chance pair.
        If a separation is within this distance, and a space distance
        condition is satisfied, the pair is considered a chance pair. Default is
        0, or no distance, i.e. the times must be exactly the same.

    Returns
    -------
    indices1 : np.ndarray[int]
        Indices within the first set of arrays that lead to chance pair.
    indices2 : np.ndarray[int]
        Indices within the second set of arrays that lead to chance pair.

    Examples
    --------
    In the following example, we load the GLAD dataset, extract the first
    two trajectories, and find between these the array indices that satisfy
    the chance pair criteria of 6 km separation distance and no time separation:

    >>> from clouddrift.datasets import glad
    >>> from clouddrift.pairs import chance_pair
    >>> from clouddrift.ragged import unpack
    >>> ds = glad()
    >>> lon1 = unpack(ds["longitude"], ds["rowsize"], rows=0).pop()
    >>> lat1 = unpack(ds["latitude"], ds["rowsize"], rows=0).pop()
    >>> time1 = unpack(ds["time"], ds["rowsize"], rows=0).pop()
    >>> lon2 = unpack(ds["longitude"], ds["rowsize"], rows=1).pop()
    >>> lat2 = unpack(ds["latitude"], ds["rowsize"], rows=1).pop()
    >>> time2 = unpack(ds["time"], ds["rowsize"], rows=1).pop()
    >>> i1, i2 = chance_pair(lon1, lat1, lon2, lat2, time1, time2, 6000, np.timedelta64(0))
    >>> i1, i2
    (array([177, 180, 183, 186, 189, 192]), array([166, 169, 172, 175, 178, 181]))

    Check to ensure our collocation in space worked by calculating the distance
    between the identified pairs:

    >>> sphere.distance(lon1[i1], lat1[i1], lon2[i2], lat2[i2])
    array([5967.4844, 5403.253 , 5116.9136, 5185.715 , 5467.8555, 5958.4917],
          dtype=float32)

    Check the collocation in time:

    >>> time1[i1] - time2[i2]
    <xarray.DataArray 'time' (obs: 6)>
    array([0, 0, 0, 0, 0, 0], dtype='timedelta64[ns]')
    Coordinates:
        time     (obs) datetime64[ns] 2012-07-21T21:30:00.524160 ... 2012-07-22T0...
    Dimensions without coordinates: obs

    Raises
    ------
    ValueError
        If ``time1`` and ``time2`` are not both provided or both omitted.
    """
    if (time1 is None and time2 is not None) or (time1 is not None and time2 is None):
        raise ValueError(
            "Both time1 and time2 must be provided or both must be omitted."
        )

    time_present = time1 is not None and time2 is not None

    if time_present:
        # If time is provided, subset the trajectories to the overlapping times.
        overlap1, overlap2 = pair_time_overlap(time1, time2, time_distance)
    else:
        # Otherwise, initialize the overlap indices to the full length of the
        # trajectories.
        overlap1 = np.arange(lon1.size)
        overlap2 = np.arange(lon2.size)

    # Provided space distance is in meters, but here we convert it to degrees
    # for the bounding box overlap check.
    space_distance_degrees = np.degrees(space_distance / sphere.EARTH_RADIUS_METERS)

    # Compute the indices for each trajectory where the two trajectories'
    # bounding boxes overlap.
    bbox_overlap1, bbox_overlap2 = pair_bounding_box_overlap(
        lon1[overlap1],
        lat1[overlap1],
        lon2[overlap2],
        lat2[overlap2],
        space_distance_degrees,
    )

    # bbox_overlap1 and bbox_overlap2 subset the overlap1 and overlap2 indices.
    overlap1 = overlap1[bbox_overlap1]
    overlap2 = overlap2[bbox_overlap2]

    # If time is present, first search for collocation in time.
    if time_present:
        time_separation = pair_time_distance(time1[overlap1], time2[overlap2])
        time_match2, time_match1 = np.where(time_separation <= time_distance)
        overlap1 = overlap1[time_match1]
        overlap2 = overlap2[time_match2]

    # Now search for collocation in space.
    space_separation = pair_space_distance(
        lon1[overlap1], lat1[overlap1], lon2[overlap2], lat2[overlap2]
    )
    space_overlap = space_separation <= space_distance
    if time_present:
        time_separation = pair_time_distance(time1[overlap1], time2[overlap2])
        time_overlap = time_separation <= time_distance
        match2, match1 = np.where(space_overlap & time_overlap)
    else:
        match2, match1 = np.where(space_overlap)

    overlap1 = overlap1[match1]
    overlap2 = overlap2[match2]

    return overlap1, overlap2


def chance_pairs_from_ragged(
    lon: array_like,
    lat: array_like,
    rowsize: array_like,
    space_distance: Optional[float] = 0,
    time: Optional[array_like] = None,
    time_distance: Optional[float] = 0,
) -> List[Tuple[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]]:
    """Return all chance pairs of contiguous trajectories in a ragged array,
    and their collocated points in space and (optionally) time, given input
    ragged arrays of longitude, latitude, and (optionally) time, and chance
    pair criteria as maximum allowable distances in space and time.

    If ``time`` and ``time_distance`` are omitted, the search will be done
    only on the spatial criteria, and the result will not include the time
    arrays.

    If ``time`` and ``time_distance`` are provided, the search will be done
    on both the spatial and temporal criteria, and the result will include the
    time arrays.

    Parameters
    ----------
    lon : array_like
        Array of longitudes in degrees.
    lat : array_like
        Array of latitudes in degrees.
    rowsize : array_like
        Array of rowsizes.
    space_distance : float, optional
        Maximum space distance in meters for the pair to qualify as chance pair.
        If the separation is within this distance, the pair is considered to be
        a chance pair. Default is 0, or no distance, i.e. the positions must be
        exactly the same.
    time : array_like, optional
        Array of times.
    time_distance : float, optional
        Maximum time distance allowed for the pair to qualify as chance pair.
        If the separation is within this distance, and the space distance
        condition is satisfied, the pair is considered a chance pair. Default is
        0, or no distance, i.e. the times must be exactly the same.

    Returns
    -------
    pairs : List[Tuple[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]]
        List of tuples, each tuple containing a Tuple of integer indices that
        corresponds to the trajectory rows in the ragged array, indicating the
        pair of trajectories that satisfy the chance pair criteria, and a Tuple
        of arrays containing the indices of the collocated points for each
        trajectory in the chance pair.

    Examples
    --------
    In the following example, we load GLAD dataset as a ragged array dataset,
    subset the result to retain the first five trajectories, and finally find
    all trajectories that satisfy the chance pair criteria of 12 km separation
    distance and no time separation, as well as the indices of the collocated
    points for each pair.

    >>> from clouddrift.datasets import glad
    >>> from clouddrift.pairs import chance_pairs_from_ragged
    >>> from clouddrift.ragged import subset
    >>> ds = subset(glad(), {"id": ["CARTHE_001", "CARTHE_002", "CARTHE_003", "CARTHE_004", "CARTHE_005"]}, id_var_name="id")
    >>> pairs = chance_pairs_from_ragged(
        ds["longitude"].values,
        ds["latitude"].values,
        ds["rowsize"].values,
        space_distance=12000,
        time=ds["time"].values,
        time_distance=np.timedelta64(0)
    )
    [((0, 1),
      (array([153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189,
              192, 195, 198, 201, 204, 207, 210, 213, 216]),
       array([142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178,
              181, 184, 187, 190, 193, 196, 199, 202, 205]))),
     ((3, 4),
      (array([141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177,
              180, 183]),
       array([136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172,
              175, 178])))]

    The result above shows that 2 chance pairs were found.

    Raises
    ------
    ValueError
        If ``rowsize`` has fewer than two elements.
    """
    if len(rowsize) < 2:
        raise ValueError("rowsize must have at least two elements.")
    pairs = list(itertools.combinations(np.arange(rowsize.size), 2))
    i = ragged.rowsize_to_index(rowsize)
    results = []
    with ThreadPoolExecutor() as executor:
        if time is None:
            futures = [
                executor.submit(
                    chance_pair,
                    lon[i[j] : i[j + 1]],
                    lat[i[j] : i[j + 1]],
                    lon[i[k] : i[k + 1]],
                    lat[i[k] : i[k + 1]],
                    space_distance=space_distance,
                )
                for j, k in pairs
            ]
        else:
            futures = [
                executor.submit(
                    chance_pair,
                    lon[i[j] : i[j + 1]],
                    lat[i[j] : i[j + 1]],
                    lon[i[k] : i[k + 1]],
                    lat[i[k] : i[k + 1]],
                    time[i[j] : i[j + 1]],
                    time[i[k] : i[k + 1]],
                    space_distance,
                    time_distance,
                )
                for j, k in pairs
            ]
        for future in as_completed(futures):
            res = future.result()
            # chance_pair function returns empty arrays if no chance criteria
            # are satisfied. We only want to keep pairs that satisfy the
            # criteria. chance_pair returns a tuple of arrays that are always
            # the same size, so we only need to check the length of the first
            # array.
            if res[0].size > 0:
                results.append((pairs[futures.index(future)], res))
    return results


def pair_bounding_box_overlap(
    lon1: array_like,
    lat1: array_like,
    lon2: array_like,
    lat2: array_like,
    distance: Optional[float] = 0,
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
    distance : float, optional
        Distance in degrees for the overlap. If the overlap is within this
        distance, the bounding boxes are considered to overlap. Default is 0.

    Returns
    -------
    overlap1 : np.ndarray[int]
        Indices ``lon1`` and ``lat1`` where their bounding box overlaps with
        that of ``lon2`` and ``lat2``.
    overlap2 : np.ndarray[int]
        Indices ``lon2`` and ``lat2`` where their bounding box overlaps with
        that of ``lon1`` and ``lat1``.

    Examples
    --------
    >>> lon1 = [0, 0, 1, 1]
    >>> lat1 = [0, 0, 1, 1]
    >>> lon2 = [1, 1, 2, 2]
    >>> lat2 = [1, 1, 2, 2]
    >>> pair_bounding_box_overlap(lon1, lat1, lon2, lat2, 0.5)
    (array([2, 3]), array([0, 1]))
    """
    # First get the bounding box of each trajectory.
    # We unwrap the longitudes before computing min/max because we want to
    # consider trajectories that cross the dateline.
    lon1_min, lon1_max = (
        np.min(np.unwrap(lon1, period=360)),
        np.max(np.unwrap(lon1, period=360)),
    )
    lat1_min, lat1_max = np.min(lat1), np.max(lat1)
    lon2_min, lon2_max = (
        np.min(np.unwrap(lon2, period=360)),
        np.max(np.unwrap(lon2, period=360)),
    )
    lat2_min, lat2_max = np.min(lat2), np.max(lat2)

    bounding_boxes_overlap = (
        (lon1_min <= lon2_max + distance)
        & (lon1_max >= lon2_min - distance)
        & (lat1_min <= lat2_max + distance)
        & (lat1_max >= lat2_min - distance)
    )

    # Now check if the trajectories overlap within the bounding box.
    if bounding_boxes_overlap:
        overlap_start = (
            max(lon1_min, lon2_min) - distance,  # West
            max(lat1_min, lat2_min) - distance,  # South
        )
        overlap_end = (
            min(lon1_max, lon2_max) + distance,  # East
            min(lat1_max, lat2_max) + distance,  # North
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
        return np.where(overlap1)[0], np.where(overlap2)[0]
    else:
        return np.array([], dtype=int), np.array([], dtype=int)


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
    quantity), return the temporal distance between all pairs of times.

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
    distance: Optional[float] = 0,
) -> Tuple[np.ndarray[int], np.ndarray[int]]:
    """Given two arrays of times (or any other monotonically increasing
    quantity), return indices where the times are within a prescribed distance.

    Although higher-level array containers like xarray and pandas are supported
    for input arrays, this function is an order of magnitude faster when passing
    in numpy arrays.

    Parameters
    ----------
    time1 : array_like
        First array of times.
    time2 : array_like
        Second array of times.
    distance : float
        Maximum distance within which the values of ``time1`` and ``time2`` are
        considered to overlap. Default is 0, or, the values must be exactly the
        same.

    Returns
    -------
    overlap1 : np.ndarray[int]
        Indices of ``time1`` where its time overlaps with ``time2``.
    overlap2 : np.ndarray[int]
        Indices of ``time2`` where its time overlaps with ``time1``.

    Examples
    --------
    >>> time1 = np.arange(4)
    >>> time2 = np.arange(2, 6)
    >>> pair_time_overlap(time1, time2)
    (array([2, 3]), array([0, 1]))

    >>> pair_time_overlap(time1, time2, 1)
    (array([1, 2, 3]), array([0, 1, 2]))
    """
    time1_min, time1_max = np.min(time1), np.max(time1)
    time2_min, time2_max = np.min(time2), np.max(time2)
    overlap_start = max(time1_min, time2_min) - distance
    overlap_end = min(time1_max, time2_max) + distance
    overlap1 = np.where((time1 >= overlap_start) & (time1 <= overlap_end))[0]
    overlap2 = np.where((time2 >= overlap_start) & (time2 <= overlap_end))[0]
    return overlap1, overlap2

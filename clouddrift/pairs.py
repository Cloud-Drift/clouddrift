"""
Functions to analyze pairs of contiguous data segments.
"""
from clouddrift import ragged, sphere
from concurrent.futures import as_completed, ThreadPoolExecutor
import itertools
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Optional, Tuple, Union

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
    """Given two arrays of longitudes and latitudes, return the chance of
    finding a pair of points that are within a given distance in space and time.

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
        Maximum space distance in meters for the pair to qualify as chance pair.
        If the separation is within this distance, the pair is considered to be
        a chance pair. Default is 0, or no distance, i.e. the positions must be
        exactly the same.
    time_distance : float, optional
        Maximum time distance allowed for the pair to qualify as chance pair.
        If the separation is within this distance, and the space distance
        condition is satisfied, the pair is considered a chance pair. Default is
        0, or no distance, i.e. the times must be exactly the same.

    Returns
    -------
    indices1 : np.ndarray[int]
        Indices of the first array that satisfy chance pair criteria.
    indices2 : np.ndarray[int]
        Indices of the second array that satisfy chance pair criteria.

    Examples
    --------
    In the following example, we load the first two trajectories from the GLAD
    dataset and find all longitudes, latitudes, and times that satisfy the
    chance pair criteria of 6 km separation distance and no time separation:

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
    >>> chance_pair(lon1, lat1, lon2, lat2, time1, time2, 6000, np.timedelta64(0))
    (array([-87.05255 , -87.04974 , -87.04612 , -87.04155 , -87.03718 ,
            -87.033165], dtype=float32),
     array([28.23184 , 28.227087, 28.222498, 28.217508, 28.2118  , 28.205938],
           dtype=float32),
     array([-87.111305, -87.10078 , -87.090675, -87.08157 , -87.07363 ,
            -87.067375], dtype=float32),
     array([28.217918, 28.20882 , 28.198603, 28.187078, 28.174644, 28.161713],
           dtype=float32),
     array(['2012-07-21T21:30:00.524160000', '2012-07-21T22:15:00.532800000',
            '2012-07-21T23:00:00.541440000', '2012-07-21T23:45:00.550080000',
            '2012-07-22T00:30:00.558720000', '2012-07-22T01:15:00.567360000'],
           dtype='datetime64[ns]'),
     array(['2012-07-21T21:30:00.524160000', '2012-07-21T22:15:00.532800000',
            '2012-07-21T23:00:00.541440000', '2012-07-21T23:45:00.550080000',
            '2012-07-22T00:30:00.558720000', '2012-07-22T01:15:00.567360000'],
           dtype='datetime64[ns]'))
    """
    time_present = time1 is not None and time2 is not None

    # If time is provided, subset the trajectories to the overlapping times.
    if time_present:
        overlap1, overlap2 = pair_time_overlap(time1, time2, time_distance)
        lon1 = lon1[overlap1]
        lat1 = lat1[overlap1]
        time1 = time1[overlap1]
        lon2 = lon2[overlap2]
        lat2 = lat2[overlap2]
        time2 = time2[overlap2]

    space_distance_degrees = np.degrees(space_distance / sphere.EARTH_RADIUS_METERS)

    bbox_overlap1, bbox_overlap2 = pair_bounding_box_overlap(
        lon1, lat1, lon2, lat2, space_distance_degrees
    )

    lon1 = lon1[bbox_overlap1]
    lat1 = lat1[bbox_overlap1]
    lon2 = lon2[bbox_overlap2]
    lat2 = lat2[bbox_overlap2]
    time1 = time1[bbox_overlap1] if time_present else None
    time2 = time2[bbox_overlap2] if time_present else None

    chance_mask = pair_space_distance(lon1, lat1, lon2, lat2) <= space_distance

    if time_present:
        time_distance = pair_time_distance(time1, time2)
        chance_mask_time = time_distance <= time_distance
        chance_mask = chance_mask & chance_mask_time

    indices2, indices1 = np.where(chance_mask)

    if time_present:
        return (
            lon1[indices1],
            lat1[indices1],
            lon2[indices2],
            lat2[indices2],
            time1[indices1],
            time2[indices2],
        )
    else:
        return lon1[indices1], lat1[indices1], lon2[indices2], lat2[indices2]


def chance_pairs(
    lon: array_like,
    lat: array_like,
    rowsize: array_like,
    space_distance: Optional[float] = 0,
    time: Optional[array_like] = None,
    time_distance: Optional[float] = 0,
) -> List[
    Tuple[
        Tuple[int, int],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
]:
    """Return all longitudes, latitudes, and (optionally) times, at which pairs
    of contiguous trajectories satisfy the chance pair criteria.

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
    pairs : List[Tuple[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
        List of tuples, each tuple being a tuple of pair indices and longitudes,
        latitudes, and times that satisfy the chance pair criteria, for each
        trajectory. The arrays are returned in the following order:
        longitude and latitude for the first trajectory, longitude and latitude
        for the second trajectory, and then times for the first and second
        trajectories, if time was provided as the input.

    Examples
    --------
    In the following example, we load the first five trajectories from the GLAD
    dataset as a ragged-array, and find all longitudes, latitudes, and times
    that satisfy the chance pair criteria of 6 km separation distance and no
    time separation:

    >>> from clouddrift.datasets import glad
    >>> from clouddrift.datasets import glad
    >>> from clouddrift.pairs import chance_pairs
    >>> from clouddrift.ragged import subset
    >>> ds = subset(glad(), {"id": ["CARTHE_001", "CARTHE_002", "CARTHE_003", "CARTHE_004", "CARTHE_005"]}, id_var_name="id")
    >>> res = chance_pairs(
        ds["longitude"].values,
        ds["latitude"].values,
        ds["rowsize"].values,
        space_distance=6000,
        time=time,
        time_distance=np.timedelta64(0)
    )
    [((0, 1),
      (array([-87.05255 , -87.04974 , -87.04612 , -87.04155 , -87.03718 ,
              -87.033165], dtype=float32),
       array([28.23184 , 28.227087, 28.222498, 28.217508, 28.2118  , 28.205938],
             dtype=float32),
       array([-87.111305, -87.10078 , -87.090675, -87.08157 , -87.07363 ,
              -87.067375], dtype=float32),
       array([28.217918, 28.20882 , 28.198603, 28.187078, 28.174644, 28.161713],
             dtype=float32),
       array(['2012-07-21T21:30:00.524160000', '2012-07-21T22:15:00.532800000',
              '2012-07-21T23:00:00.541440000', '2012-07-21T23:45:00.550080000',
              '2012-07-22T00:30:00.558720000', '2012-07-22T01:15:00.567360000'],
             dtype='datetime64[ns]'),
       array(['2012-07-21T21:30:00.524160000', '2012-07-21T22:15:00.532800000',
              '2012-07-21T23:00:00.541440000', '2012-07-21T23:45:00.550080000',
              '2012-07-22T00:30:00.558720000', '2012-07-22T01:15:00.567360000'],
             dtype='datetime64[ns]')))]

    To query only the trajectory pair indices that satisfy the chance pair
    criteria, use a list comprehension. The following example returns the
    trajectory pair indices that satisfy the chance pair criteria of 60 km:

    >>> res = chance_pairs(
        ds["longitude"].values,
        ds["latitude"].values,
        ds["rowsize"].values,
        space_distance=60000,
        time=time,
        time_distance=np.timedelta64(0)
    )
    >>> [p[0] for p in res]
    [(1, 4), (1, 3), (2, 3), (0, 1), (2, 4), (1, 2), (0, 2), (0, 3), (3, 4)]
    """
    # TODO check len(rowsize) >= set_size
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
    lon1_min, lon1_max = np.min(lon1), np.max(lon1)
    lat1_min, lat1_max = np.min(lat1), np.max(lat1)
    lon2_min, lon2_max = np.min(lon2), np.max(lon2)
    lat2_min, lat2_max = np.min(lat2), np.max(lat2)

    # TODO handle trajectories that cross the dateline

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
    distance: Optional[float] = 0,
) -> Tuple[np.ndarray[int], np.ndarray[int]]:
    """Given two arrays of times (or any other monotonically increasing
    quantity), return indices where the times are within a requested distance.

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

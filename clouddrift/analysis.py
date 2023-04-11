import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
import pandas as pd
from concurrent import futures
from datetime import timedelta
import warnings
from clouddrift.haversine import distance, bearing
from clouddrift.dataformat import unpack_ragged


def apply_ragged(
    func: callable,
    arrays: list[np.ndarray],
    rowsize: list[int],
    *args: tuple,
    max_workers: int = None,
    **kwargs: dict,
) -> Union[tuple[np.ndarray], np.ndarray]:
    """Apply a function to a ragged array.

    The function ``func`` will be applied to each contiguous row of ``arrays`` as
    indicated by row sizes ``rowsize``. The output of ``func`` will be
    concatenated into a single ragged array.

    This function uses ``concurrent.futures.ThreadPoolExecutor`` to run ``func``
    in multiple threads. The number of threads can be controlled by the
    ``max_workers`` argument, which is passed down to ``ThreadPoolExecutor``.

    Parameters
    ----------
    func : callable
        Function to apply to each row of each ragged array in ``arrays``.
    arrays : list[np.ndarray] or np.ndarray
        An array or a list of arrays to apply ``func`` to.
    rowsize : list
        List of integers specifying the number of data points in each row.
    *args : tuple
        Additional arguments to pass to ``func``.
    max_workers : int, optional
        Number of threads to use. If None, the number of threads will be equal
        to the ``max_workers`` default value of ``concurrent.futures.ThreadPoolExecutor``.
    **kwargs : dict
        Additional keyword arguments to pass to ``func``.

    Returns
    -------
    out : tuple[np.ndarray] or np.ndarray
        Output array(s) from ``func``.

    Examples
    --------
    >>> rowsize = [2, 3, 4]
    >>> x = np.array([1, 2, 10, 12, 14, 30, 33, 36, 39])
    >>> y = np.arange(0, len(x))
    >>> t = np.array([1, 2, 1, 2, 3, 1, 2, 3, 4])

    Using ``velocity_from_position`` with ``apply_ragged``, the velocities of each trajectory
    are obtained from the positions and time ragged arrays [x,y,t]. Note that the first trajectory
    has 2 data points, the second has 3, and the third has 4.

    >>> u1, v1 = apply_ragged(velocity_from_position, rowsize, [x, y, t], coord_system="cartesian")
    array([1., 1., 2., 2., 2., 3., 3., 3., 3.]),
    array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    Raises
    ------
    ValueError
        If the sum of ``rowsize`` does not equal the length of ``arrays``.
    """
    # make sure the arrays is iterable
    if type(arrays) not in [list, tuple]:
        arrays = [arrays]
    # validate rowsize
    for arr in arrays:
        if not sum(rowsize) == len(arr):
            raise ValueError("The sum of rowsize must equal the length of arr.")

    # split the array(s) into trajectories
    arrays = [unpack_ragged(arr, rowsize) for arr in arrays]
    iter = [[arrays[i][j] for i in range(len(arrays))] for j in range(len(arrays[0]))]

    # combine other arguments
    for arg in iter:
        if args:
            arg.append(*args)

    # parallel execution
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        res = executor.map(lambda x: func(*x, **kwargs), iter)
    # concatenate the outputs
    res = list(res)
    if isinstance(res[0], tuple):  # more than 1 parameter
        outputs = []
        for i in range(len(res[0])):
            outputs.append(np.concatenate([r[i] for r in res]))
        return tuple(outputs)
    else:
        return np.concatenate(res)


def chunk(
    x: Union[list, np.ndarray, xr.DataArray, pd.Series],
    length: int,
    overlap: int = 0,
    align: str = "start",
) -> np.ndarray:
    """Chunk an array ``x`` into equal-length chunks while respecting
    the contiguous segments of the ragged array. The result is 2-dimensional
    NumPy array of shape ``(num_chunks, length)``. The resulting number of chunks
    is determined based on the length of ``x``, ``length``, and ``overlap``.

    ``chunk`` is compatible with :func:`apply_ragged`, which allows you to chunk
    a ragged array.

    Parameters
    ----------
    x : list or array-like
        Array to split into chunks.
    length : int
        The length of each chunk.
    overlap : int, optional
        The number of overlapping points between chunks. The default is 0.
        Must be smaller than ``length``. For example, if ``length`` is 4 and
        ``overlap`` is 2, the chunks of ``[0, 1, 2, 3, 4, 5]`` will be
        ``np.array([[0, 1, 2, 3], [2, 3, 4, 5]])``. Negative overlap can be used
        to offset chunks by some number of elements. For example, if ``length``
        is 2 and ``overlap`` is -1, the chunks of ``[0, 1, 2, 3, 4, 5]`` will
        be ``np.array([[0, 1], [3, 4]])``.
    align : str, optional ["start", "middle", "end"]
        If the number of chunks (including or not overlap) is not a multiple of the
        length of ``x`` and there is a reminder of N points, this parameter controls
        which part of the array will be kept into the chunks. If ``align="start"``, the
        points at the beginning of the array will be kept, and N points are discarded at
        the end. If `align="middle"`, floor(N/2) and ceil(N/2) points will be respectively
        discarded from the beginning and the end of the array. If ``align="end"``, the
        points at the end of the array will be kept, and the `N` first points are discarded.
        The default is "start".

    Returns
    -------
    np.ndarray
        2-dimensional array of shape ``(num_chunks, length)``.

    Examples
    --------

    Chunk a simple list; this will trim the end that exceeds the last chunk:

    >>> chunk([1, 2, 3, 4, 5], 2)
    array([[1, 2],
           [3, 4]])

    To trim the beginning of the array, use ``align="end"``:
    >>> chunk([1, 2, 3, 4, 5], 2, align="end")
    array([[2, 3],
           [4, 5]])

    or to centered the chunks with respect to the array, use ``align="middle"``:
    >>> chunk([1, 2, 3, 4, 5, 6, 7, 8], 3, align="middle")
    array([[2, 3, 4],
           [5, 6, 7]])

    Specify ``overlap`` to get overlapping chunks:

    >>> chunk([1, 2, 3, 4, 5], 2, overlap=1)
    array([[1, 2],
           [2, 3],
           [3, 4],
           [4, 5]])

    Use ``apply_ragged`` to chunk a ragged array; notice that you must pass the
    array to chunk as an array-like, not a list:

    >>> apply_ragged(chunk, np.array([1, 2, 3, 4, 5]), rowsize=[2, 1, 2], 2)])
    array([[1, 2],
           [4, 5]])

    Raises
    ------
    ValueError
        If ``length < 0``.
    ValueError
        If ``align not in ["start", "middle", "end"]``.
    ZeroDivisionError
        if ``length == 0``.
    """
    num_chunks = (len(x) - length) // (length - overlap) + 1 if len(x) >= length else 0
    remainder = len(x) - num_chunks * length + (num_chunks - 1) * overlap
    res = np.empty((num_chunks, length), dtype=np.array(x).dtype)

    if align == "start":
        start = 0
    elif align == "middle":
        start = remainder // 2
    elif align == "end":
        start = remainder
    else:
        raise ValueError("align must be one of 'start', 'middle', or 'end'.")

    for n in range(num_chunks):
        end = start + length
        res[n] = x[start:end]
        start = end - overlap

    return res


def regular_to_ragged(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a two-dimensional array to a ragged array. NaN values in the input array are
    excluded from the output ragged array.

    Parameters
    ----------
    array : np.ndarray
        A two-dimensional array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of the ragged array and the size of each row.

    Examples
    --------
    >>> regular_to_ragged(np.array([[1, 2], [3, np.nan], [4, 5]]))
    (array([1., 2., 3., 4., 5.]), array([2, 1, 2]))

    See Also
    --------
    :func:`ragged_to_regular`
    """
    ragged = array.flatten()
    return ragged[~np.isnan(ragged)], np.sum(~np.isnan(array), axis=1)


def ragged_to_regular(
    ragged: Union[np.ndarray, pd.Series, xr.DataArray],
    rowsize: Union[list, np.ndarray, pd.Series, xr.DataArray],
) -> np.ndarray:
    """Convert a ragged array to a two-dimensional array such that each contiguous segment
    of a ragged array is a row in the two-dimensional array, and the remaining elements are
    padded with NaNs.

    Note: Although this function accepts parameters of type ``xarray.DataArray``,
    passing NumPy arrays is recommended for performance reasons.

    Parameters
    ----------
    ragged : np.ndarray or pd.Series or xr.DataArray
        A ragged array.
    rowsize : list or np.ndarray[int] or pd.Series or xr.DataArray[int]
        The size of each row in the ragged array.

    Returns
    -------
    np.ndarray
        A two-dimensional array.

    Examples
    --------
    >>> ragged_to_regular(np.array([1, 2, 3, 4, 5]), np.array([2, 1, 2]))
    array([[ 1.,  2.],
           [ 3., nan],
           [ 4.,  5.]])

    See Also
    --------
    :func:`regular_to_ragged`
    """
    res = np.nan * np.empty((len(rowsize), int(max(rowsize))), dtype=ragged.dtype)
    unpacked = unpack_ragged(ragged, rowsize)
    for n in range(len(rowsize)):
        res[n, : int(rowsize[n])] = unpacked[n]
    return res


def segment(
    x: np.ndarray,
    tolerance: Union[float, np.timedelta64, timedelta, pd.Timedelta],
    rowsize: np.ndarray[int] = None,
) -> np.ndarray[int]:
    """Segment an array into contiguous segments.

    Parameters
    ----------
    x : list, np.ndarray, or xr.DataArray
        An array to segment.
    tolerance : float, np.timedelta64, timedelta, pd.Timedelta
        The maximum signed difference between consecutive points in a segment.
    rowsize : np.ndarray[int], optional
        The size of rows if x is a ragged array. If present, x will be
        segmented both by gaps that exceed the tolerance, and by rows
        of the ragged array.

    Returns
    -------
    segment_sizes : np.ndarray[int]
        An array of row-sizes that segment the input array into contiguous segments.

    Examples
    --------

    The simplest use of ``segment`` is to provide a tolerance value that is
    used to segment an array into contiguous segments.

    >>> x = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
    >>> segment(x, 0.5)
    array([1, 3, 2, 4, 1])

    If the array represents time and the tolerance is a timedelta,
    the same logic applies.

    >>> x = np.array([np.datetime64("2023-01-01"), np.datetime64("2023-01-02"),
                      np.datetime64("2023-01-03"), np.datetime64("2023-02-01"),
                      np.datetime64("2023-02-02")])
    >>> segment(x, np.timedelta64(1, "D"))
    np.array([3, 2])

    If the array is already previously segmented (e.g. multiple trajectories
    as a ragged array), then the ``rowsize`` argument can be used to preserve
    the input segments.

    >>> rowsize = [3, 2, 6]
    >>> segment(x, 0.5, rowsize)
    array([1, 2, 1, 1, 1, 4, 1])

    The tolerance can also be negative. In this case, the segments are
    determined by the gaps where the negative difference exceeds the negative
    value of the tolerance, i.e. where ``x[n+1] - x[n] < -tolerance``.

    >>> x = [0, 1, 2, 0, 1, 2]
    >>> segment(x, -0.5)
    array([3, 3])

    To segment an array for both positive and negative gaps, invoke the function
    twice, once for a positive tolerance and once for a negative tolerance.
    The result of the first invocation can be passed as the ``rowsize`` argument
    to the first ``segment`` invocation.

    >>> x = [1, 1, 2, 2, 1, 1, 2, 2]
    >>> segment(x, 0.5, rowsize=segment(x, -0.5))
    array([2, 2, 2, 2])
    """

    # for compatibility with datetime list or np.timedelta64 arrays
    if type(tolerance) in [np.timedelta64, timedelta]:
        tolerance = pd.Timedelta(tolerance)

    if type(tolerance) == pd.Timedelta:
        positive_tol = tolerance >= pd.Timedelta("0 seconds")
    else:
        positive_tol = tolerance >= 0

    if rowsize is None:
        if positive_tol:
            exceeds_tolerance = np.diff(x) > tolerance
        else:
            exceeds_tolerance = np.diff(x) < tolerance
        segment_sizes = np.diff(np.insert(np.where(exceeds_tolerance)[0] + 1, 0, 0))
        segment_sizes = np.append(segment_sizes, len(x) - np.sum(segment_sizes))
        return segment_sizes
    else:
        if not sum(rowsize) == len(x):
            raise ValueError("The sum of rowsize must equal the length of x.")
        segment_sizes = []
        start = 0
        for r in rowsize:
            end = start + int(r)
            segment_sizes.append(segment(x[start:end], tolerance))
            start = end
        return np.concatenate(segment_sizes)


def velocity_from_position(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    coord_system: Optional[str] = "spherical",
    difference_scheme: Optional[str] = "forward",
    time_axis: Optional[int] = -1,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute velocity from arrays of positions and time.

    x and y can be provided as longitude and latitude in degrees if
    coord_system == "spherical" (default), or as easting and northing if
    coord_system == "cartesian".

    The units of the result are meters per unit of time if
    coord_system == "spherical". For example, if the time is provided in the
    units of seconds, the resulting velocity is in the units of meters per
    second. Otherwise, if coord_system == "cartesian", the units of the
    resulting velocity correspond to the units of the input. For example,
    if Easting and Northing are in the units of kilometers and time is in
    the units of hours, the resulting velocity is in the units of kilometers
    per hour.

    x, y, and time can be multi-dimensional arrays. If the time axis, along
    which the finite differencing is performed, is not the last one (i.e.
    x.shape[-1]), use the time_axis optional argument to specify along which
    axis should the differencing be done. x, y, and time must have the same
    shape.

    Difference scheme can take one of three values:

        1. "forward" (default): finite difference is evaluated as
           dx[i] = dx[i+1] - dx[i];
        2. "backward": finite difference is evaluated as
           dx[i] = dx[i] - dx[i-1];
        3. "centered": finite difference is evaluated as
           dx[i] = (dx[i+1] - dx[i-1]) / 2.

    Forward and backward schemes are effectively the same except that the
    position at which the velocity is evaluated is shifted one element down in
    the backward scheme relative to the forward scheme. In the case of a
    forward or backward difference scheme, the last or first element of the
    velocity, respectively, is extrapolated from its neighboring point. In the
    case of a centered difference scheme, the start and end boundary points are
    evaluated using the forward and backward difference scheme, respectively.

    Args:
        x (array_like): An N-d array of x-positions (longitude in degrees or easting in any unit)
        y (array_like): An N-d array of y-positions (latitude in degrees or northing in any unit)
        time (array_like): An N-d array of times as floating point values (in any unit)
        coord_system (str, optional): Coordinate system that x and y arrays are in; possible values are "spherical" (default) or "cartesian".
        difference_scheme (str, optional): Difference scheme to use; possible values are "forward", "backward", and "centered".
        time_axis (int, optional): Axis along which to differentiate (default is -1)

    Returns:
        out (Tuple[xr.DataArray[float], xr.DataArray[float]]): Arrays of x- and y-velocities
    """

    # Positions and time arrays must have the same shape.
    if not x.shape == y.shape == time.shape:
        raise ValueError("x, y, and time must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(x.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )

    # Nominal order of axes on input, i.e. (0, 1, 2, ..., N-1)
    target_axes = list(range(len(x.shape)))

    # If time_axis is not the last one, transpose the inputs
    if time_axis != -1 and time_axis < len(x.shape) - 1:
        target_axes.append(target_axes.pop(target_axes.index(time_axis)))

    # Reshape the inputs to ensure the time axis is last (fast-varying)
    x_ = np.transpose(x, target_axes)
    y_ = np.transpose(y, target_axes)
    time_ = np.transpose(time, target_axes)

    dx = np.empty(x_.shape)
    dy = np.empty(y_.shape)
    dt = np.empty(time_.shape)

    # Compute dx, dy, and dt
    if difference_scheme == "forward":
        # All values except the ending boundary value are computed using the
        # 1st order forward differencing. The ending boundary value is
        # computed using the 1st order backward difference.

        # Time
        dt[..., :-1] = np.diff(time_)
        dt[..., -1] = dt[..., -2]

        # Space
        if coord_system == "cartesian":
            dx[..., :-1] = np.diff(x_)
            dx[..., -1] = dx[..., -2]
            dy[..., :-1] = np.diff(y_)
            dy[..., -1] = dy[..., -2]

        elif coord_system == "spherical":
            distances = distance(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
            bearings = bearing(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
            dx[..., :-1] = distances * np.cos(bearings)
            dx[..., -1] = dx[..., -2]
            dy[..., :-1] = distances * np.sin(bearings)
            dy[..., -1] = dy[..., -2]

        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    elif difference_scheme == "backward":
        # All values except the starting boundary value are computed using the
        # 1st order backward differencing. The starting boundary value is
        # computed using the 1st order forward difference.

        # Time
        dt[..., 1:] = np.diff(time_)
        dt[..., 0] = dt[..., 1]

        # Space
        if coord_system == "cartesian":
            dx[..., 1:] = np.diff(x_)
            dx[..., 0] = dx[..., 1]
            dy[..., 1:] = np.diff(y_)
            dy[..., 0] = dy[..., 1]

        elif coord_system == "spherical":
            distances = distance(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
            bearings = bearing(y_[..., :-1], x_[..., :-1], y_[..., 1:], x_[..., 1:])
            dx[..., 1:] = distances * np.cos(bearings)
            dx[..., 0] = dx[..., 1]
            dy[..., 1:] = distances * np.sin(bearings)
            dy[..., 0] = dy[..., 1]

        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    elif difference_scheme == "centered":
        # Inner values are computed using the 2nd order centered differencing.
        # The start and end boundary values are computed using the 1st order
        # forward and backward differencing, respectively.

        # Time
        dt[..., 1:-1] = (time_[..., 2:] - time_[..., :-2]) / 2
        dt[..., 0] = time_[..., 1] - time_[..., 0]
        dt[..., -1] = time_[..., -1] - time_[..., -2]

        # Space
        if coord_system == "cartesian":
            dx[..., 1:-1] = (x_[..., 2:] - x_[..., :-2]) / 2
            dx[..., 0] = x_[..., 1] - x_[..., 0]
            dx[..., -1] = x_[..., -1] - x_[..., -2]
            dy[..., 1:-1] = (y_[..., 2:] - y_[..., :-2]) / 2
            dy[..., 0] = y_[..., 1] - y_[..., 0]
            dy[..., -1] = y_[..., -1] - y_[..., -2]

        elif coord_system == "spherical":
            # Inner values
            distances = distance(y_[..., :-2], x_[..., :-2], y_[..., 2:], x_[..., 2:])
            bearings = bearing(y_[..., :-2], x_[..., :-2], y_[..., 2:], x_[..., 2:])
            dx[..., 1:-1] = distances * np.cos(bearings) / 2
            dy[..., 1:-1] = distances * np.sin(bearings) / 2

            # Boundary values
            distance1 = distance(y_[..., 0], x_[..., 0], y_[..., 1], x_[..., 1])
            bearing1 = bearing(y_[..., 0], x_[..., 0], y_[..., 1], x_[..., 1])
            dx[..., 0] = distance1 * np.cos(bearing1)
            dy[..., 0] = distance1 * np.sin(bearing1)
            distance2 = distance(y_[..., -2], x_[..., -2], y_[..., -1], x_[..., -1])
            bearing2 = bearing(y_[..., -2], x_[..., -2], y_[..., -1], x_[..., -1])
            dx[..., -1] = distance2 * np.cos(bearing2)
            dy[..., -1] = distance2 * np.sin(bearing2)

        else:
            raise ValueError('coord_system must be "spherical" or "cartesian".')

    else:
        raise ValueError(
            'difference_scheme must be "forward", "backward", or "centered".'
        )

    if target_axes == list(range(len(x.shape))):
        return dx / dt, dy / dt
    else:
        return np.transpose(dx / dt, target_axes), np.transpose(dy / dt, target_axes)


def mask_var(
    var: xr.DataArray, criterion: Union[tuple, list, bool, float, int]
) -> xr.DataArray:
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
    """Subset the dataset as a function of one or many criteria. The criteria are
    passed as a dictionary, where a variable to subset is assigned to either a
    range (valuemin, valuemax), a list [value1, value2, valueN], or a single value.

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

"""
This module provides common Lagrangian analysis and transformation
functions.
"""

import numpy as np
from typing import Optional, Tuple, Union, Iterable
import xarray as xr
import pandas as pd
from concurrent import futures
from datetime import timedelta
import warnings
from clouddrift.sphere import distance, bearing, position_from_distance_and_bearing


def apply_ragged(
    func: callable,
    arrays: list[np.ndarray],
    count: list[int],
    *args: tuple,
    executor: futures.Executor = futures.ThreadPoolExecutor(max_workers=None),
    **kwargs: dict,
) -> Union[tuple[np.ndarray], np.ndarray]:
    """Apply a function to a ragged array.

    The function ``func`` will be applied to each contiguous row of ``arrays`` as
    indicated by row sizes ``count``. The output of ``func`` will be
    concatenated into a single ragged array.

    By default this function uses ``concurrent.futures.ThreadPoolExecutor`` to
    run ``func`` in multiple threads. The number of threads can be controlled by
    passing the ``max_workers`` argument to the executor instance passed to
    ``apply_ragged``. Alternatively, you can pass the ``concurrent.futures.ProcessPoolExecutor``
    instance to use processes instead. Passing alternative (3rd party library)
    concurrent executors may work if they follow the same executor interface as
    that of ``concurrent.futures``, however this has not been tested yet.

    Parameters
    ----------
    func : callable
        Function to apply to each row of each ragged array in ``arrays``.
    arrays : list[np.ndarray] or np.ndarray
        An array or a list of arrays to apply ``func`` to.
    count : list
        List of integers specifying the number of data points in each row.
    *args : tuple
        Additional arguments to pass to ``func``.
    executor : concurrent.futures.Executor, optional
        Executor to use for concurrent execution. Default is ``ThreadPoolExecutor``
        with the default number of ``max_workers``.
        Another supported option is ``ProcessPoolExecutor``.
    **kwargs : dict
        Additional keyword arguments to pass to ``func``.

    Returns
    -------
    out : tuple[np.ndarray] or np.ndarray
        Output array(s) from ``func``.

    Examples
    --------

    Using ``velocity_from_position`` with ``apply_ragged``, calculate the velocities of
    multiple particles, the coordinates of which are found in the ragged arrays x, y, and t
    that share row sizes 2, 3, and 4:

    >>> count = [2, 3, 4]
    >>> x = np.array([1, 2, 10, 12, 14, 30, 33, 36, 39])
    >>> y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> t = np.array([1, 2, 1, 2, 3, 1, 2, 3, 4])
    >>> u1, v1 = apply_ragged(velocity_from_position, [x, y, t], count, coord_system="cartesian")
    array([1., 1., 2., 2., 2., 3., 3., 3., 3.]),
    array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    Raises
    ------
    ValueError
        If the sum of ``count`` does not equal the length of ``arrays``.
    IndexError
        If empty ``arrays``.
    """
    # make sure the arrays is iterable
    if type(arrays) not in [list, tuple]:
        arrays = [arrays]
    # validate count
    for arr in arrays:
        if not sum(count) == len(arr):
            raise ValueError("The sum of count must equal the length of arr.")

    # split the array(s) into trajectories
    arrays = [unpack_ragged(arr, count) for arr in arrays]
    iter = [[arrays[i][j] for i in range(len(arrays))] for j in range(len(arrays[0]))]

    # parallel execution
    res = [executor.submit(func, *x, *args, **kwargs) for x in iter]
    res = [r.result() for r in res]

    # concatenate the outputs
    res = [item if isinstance(item, Iterable) else [item] for item in res]

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
    """Divide an array ``x`` into equal chunks of length ``length``. The result
    is a 2-dimensional NumPy array of shape ``(num_chunks, length)``. The resulting
    number of chunks is determined based on the length of ``x``, ``length``,
    and ``overlap``.

    ``chunk`` can be combined with :func:`apply_ragged` in order to chunk a ragged
    array.

    Parameters
    ----------
    x : list or array-like
        Array to divide into chunks.
    length : int
        The length of each chunk.
    overlap : int, optional
        The number of overlapping array elements across chunks. The default is 0.
        Must be smaller than ``length``. For example, if ``length`` is 4 and
        ``overlap`` is 2, the chunks of ``[0, 1, 2, 3, 4, 5]`` will be
        ``np.array([[0, 1, 2, 3], [2, 3, 4, 5]])``. Negative overlap can be used
        to offset chunks by some number of elements. For example, if ``length``
        is 2 and ``overlap`` is -1, the chunks of ``[0, 1, 2, 3, 4, 5]`` will
        be ``np.array([[0, 1], [3, 4]])``.
    align : str, optional ["start", "middle", "end"]
        If the remainder of the length of ``x`` divided by the chunk ``length`` is a number
        N different from zero, this parameter controls which part of the array will be kept
        into the chunks. If ``align="start"``, the elements at the beginning of the array
        will be part of the chunks and N points are discarded at the end. If `align="middle"`,
        floor(N/2) and ceil(N/2) elements will be discarded from the beginning and the end
        of the array, respectively. If ``align="end"``, the elements at the end of the array
        will be kept, and the `N` first elements are discarded. The default is "start".

    Returns
    -------
    np.ndarray
        2-dimensional array of shape ``(num_chunks, length)``.

    Examples
    --------

    Chunk a simple list; this discards the end elements that exceed the last chunk:

    >>> chunk([1, 2, 3, 4, 5], 2)
    array([[1, 2],
           [3, 4]])

    To discard the starting elements of the array instead, use ``align="end"``:
    >>> chunk([1, 2, 3, 4, 5], 2, align="end")
    array([[2, 3],
           [4, 5]])

    To center the chunks by discarding both ends of the array, use ``align="middle"``:
    >>> chunk([1, 2, 3, 4, 5, 6, 7, 8], 3, align="middle")
    array([[2, 3, 4],
           [5, 6, 7]])

    Specify ``overlap`` to get overlapping chunks:

    >>> chunk([1, 2, 3, 4, 5], 2, overlap=1)
    array([[1, 2],
           [2, 3],
           [3, 4],
           [4, 5]])

    Use ``apply_ragged`` to chunk a ragged array by providing the row sizes;
    notice that you must pass the array to chunk as an array-like, not a list:

    >>> x = np.array([1, 2, 3, 4, 5])
    >>> count = [2, 1, 2]
    >>> apply_ragged(chunk, x, count, 2)
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


def prune(
    ragged: Union[list, np.ndarray, pd.Series, xr.DataArray],
    count: Union[list, np.ndarray, pd.Series, xr.DataArray],
    min_count: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Within a ragged array, removes arrays less than a specified row size.

    Parameters
    ----------
    ragged : np.ndarray or pd.Series or xr.DataArray
        A ragged array.
    count : list or np.ndarray[int] or pd.Series or xr.DataArray[int]
        The size of each row in the input ragged array.
    min_count :
        The minimum row size that will be kept.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of ragged array and size of each row.

    Examples
    --------
    >>> prune(np.array([1, 2, 3, 0, -1, -2]), np.array([3, 1, 2]),2)
    (array([1, 2, 3, -1, -2]), array([3, 2]))

    Raises
    ------
    ValueError
        If the sum of ``count`` does not equal the length of ``arrays``.
    IndexError
        If empty ``ragged``.

    See Also
    --------
    :func:`segment`, `chunk`
    """

    ragged = apply_ragged(
        lambda x, min_len: x if len(x) >= min_len else np.empty(0, dtype=x.dtype),
        np.array(ragged),
        count,
        min_len=min_count,
    )
    count = apply_ragged(
        lambda x, min_len: x if x >= min_len else np.empty(0, dtype=x.dtype),
        np.array(count),
        np.ones_like(count),
        min_len=min_count,
    )

    return ragged, count


def regular_to_ragged(
    array: np.ndarray, fill_value: float = np.nan
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a two-dimensional array to a ragged array. Fill values in the input array are
    excluded from the output ragged array.

    Parameters
    ----------
    array : np.ndarray
        A two-dimensional array.
    fill_value : float, optional
        Fill value used to determine the bounds of contiguous segments.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of the ragged array and the size of each row.

    Examples
    --------
    By default, NaN values found in the input regular array are excluded from
    the output ragged array:

    >>> regular_to_ragged(np.array([[1, 2], [3, np.nan], [4, 5]]))
    (array([1., 2., 3., 4., 5.]), array([2, 1, 2]))

    Alternatively, a different fill value can be specified:

    >>> regular_to_ragged(np.array([[1, 2], [3, -999], [4, 5]]), fill_value=-999)
    (array([1., 2., 3., 4., 5.]), array([2, 1, 2]))

    See Also
    --------
    :func:`ragged_to_regular`
    """
    if np.isnan(fill_value):
        valid = ~np.isnan(array)
    else:
        valid = array != fill_value
    return array[valid], np.sum(valid, axis=1)


def ragged_to_regular(
    ragged: Union[np.ndarray, pd.Series, xr.DataArray],
    count: Union[list, np.ndarray, pd.Series, xr.DataArray],
    fill_value: float = np.nan,
) -> np.ndarray:
    """Convert a ragged array to a two-dimensional array such that each contiguous segment
    of a ragged array is a row in the two-dimensional array. Each row of the two-dimensional
    array is padded with NaNs as needed. The length of the first dimension of the output
    array is the length of ``count``. The length of the second dimension is the maximum
    element of ``count``.

    Note: Although this function accepts parameters of type ``xarray.DataArray``,
    passing NumPy arrays is recommended for performance reasons.

    Parameters
    ----------
    ragged : np.ndarray or pd.Series or xr.DataArray
        A ragged array.
    count : list or np.ndarray[int] or pd.Series or xr.DataArray[int]
        The size of each row in the ragged array.
    fill_value : float, optional
        Fill value to use for the trailing elements of each row of the resulting
        regular array.

    Returns
    -------
    np.ndarray
        A two-dimensional array.

    Examples
    --------
    By default, the fill value used is NaN:

    >>> ragged_to_regular(np.array([1, 2, 3, 4, 5]), np.array([2, 1, 2]))
    array([[ 1.,  2.],
           [ 3., nan],
           [ 4.,  5.]])

    You can specify an alternative fill value:
    >>> ragged_to_regular(np.array([1, 2, 3, 4, 5]), np.array([2, 1, 2]), fill_value=999)
    array([[ 1.,    2.],
           [ 3., -999.],
           [ 4.,    5.]])

    See Also
    --------
    :func:`regular_to_ragged`
    """
    res = fill_value * np.ones((len(count), int(max(count))), dtype=ragged.dtype)
    unpacked = unpack_ragged(ragged, count)
    for n in range(len(count)):
        res[n, : int(count[n])] = unpacked[n]
    return res


def segment(
    x: np.ndarray,
    tolerance: Union[float, np.timedelta64, timedelta, pd.Timedelta],
    count: np.ndarray[int] = None,
) -> np.ndarray[int]:
    """Divide an array into segments based on a tolerance value.

    Parameters
    ----------
    x : list, np.ndarray, or xr.DataArray
        An array to divide into segment.
    tolerance : float, np.timedelta64, timedelta, pd.Timedelta
        The maximum signed difference between consecutive points in a segment.
        The array x will be segmented wherever differences exceed the tolerance.
    count : np.ndarray[int], optional
        The size of rows if x is originally a ragged array. If present, x will be
        divided both by gaps that exceed the tolerance, and by the original rows
        of the ragged array.

    Returns
    -------
    np.ndarray[int]
        An array of row sizes that divides the input array into segments.

    Examples
    --------

    The simplest use of ``segment`` is to provide a tolerance value that is
    used to divide an array into segments:

    >>> x = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
    >>> segment(x, 0.5)
    array([1, 3, 2, 4, 1])

    If the array is already previously segmented (e.g. multiple rows in
    a ragged array), then the ``count`` argument can be used to preserve
    the original segments:

    >>> x = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
    >>> count = [3, 2, 6]
    >>> segment(x, 0.5, count)
    array([1, 2, 1, 1, 1, 4, 1])

    The tolerance can also be negative. In this case, the input array is
    segmented where the negative difference exceeds the negative
    value of the tolerance, i.e. where ``x[n+1] - x[n] < -tolerance``:

    >>> x = [0, 1, 2, 0, 1, 2]
    >>> segment(x, -0.5)
    array([3, 3])

    To segment an array for both positive and negative gaps, invoke the function
    twice, once for a positive tolerance and once for a negative tolerance.
    The result of the first invocation can be passed as the ``count`` argument
    to the first ``segment`` invocation:

    >>> x = [1, 1, 2, 2, 1, 1, 2, 2]
    >>> segment(x, 0.5, count=segment(x, -0.5))
    array([2, 2, 2, 2])

    If the input array contains time objects, the tolerance must be a time interval:

    >>> x = np.array([np.datetime64("2023-01-01"), np.datetime64("2023-01-02"),
                      np.datetime64("2023-01-03"), np.datetime64("2023-02-01"),
                      np.datetime64("2023-02-02")])
    >>> segment(x, np.timedelta64(1, "D"))
    np.array([3, 2])
    """

    # for compatibility with datetime list or np.timedelta64 arrays
    if type(tolerance) in [np.timedelta64, timedelta]:
        tolerance = pd.Timedelta(tolerance)

    if type(tolerance) == pd.Timedelta:
        positive_tol = tolerance >= pd.Timedelta("0 seconds")
    else:
        positive_tol = tolerance >= 0

    if count is None:
        if positive_tol:
            exceeds_tolerance = np.diff(x) > tolerance
        else:
            exceeds_tolerance = np.diff(x) < tolerance
        segment_sizes = np.diff(np.insert(np.where(exceeds_tolerance)[0] + 1, 0, 0))
        segment_sizes = np.append(segment_sizes, len(x) - np.sum(segment_sizes))
        return segment_sizes
    else:
        if not sum(count) == len(x):
            raise ValueError("The sum of count must equal the length of x.")
        segment_sizes = []
        start = 0
        for r in count:
            end = start + int(r)
            segment_sizes.append(segment(x[start:end], tolerance))
            start = end
        return np.concatenate(segment_sizes)


def position_from_velocity(
    u: np.ndarray,
    v: np.ndarray,
    time: np.ndarray,
    x_origin: float,
    y_origin: float,
    coord_system: Optional[str] = "spherical",
    integration_scheme: Optional[str] = "forward",
    time_axis: Optional[int] = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positions from arrays of velocities and time and a pair of origin
    coordinates.

    The units of the result are degrees if ``coord_system == "spherical"`` (default).
    If ``coord_system == "cartesian"``, the units of the result are equal to the
    units of the input velocities multiplied by the units of the input time.
    For example, if the input velocities are in meters per second and the input
    time is in seconds, the units of the result will be meters.

    Integration scheme can take one of three values:

        1. "forward" (default): integration from x[i] to x[i+1] is performed
            using the velocity at x[i].
        2. "backward": integration from x[i] to x[i+1] is performed using the
            velocity at x[i+1].
        3. "centered": integration from x[i] to x[i+1] is performed using the
            arithmetic average of the velocities at x[i] and x[i+1]. Note that
            this method introduces some error due to the averaging.

    u, v, and time can be multi-dimensional arrays. If the time axis, along
    which the finite differencing is performed, is not the last one (i.e.
    x.shape[-1]), use the ``time_axis`` optional argument to specify along which
    axis should the differencing be done. ``x``, ``y``, and ``time`` must have
    the same shape.

    This function will not do any special handling of longitude ranges. If the
    integrated trajectory crosses the antimeridian (dateline) in either direction, the
    longitude values will not be adjusted to stay in any specific range such
    as [-180, 180] or [0, 360]. If you need your longitudes to be in a specific
    range, recast the resulting longitude from this function using the function
    :func:`clouddrift.sphere.recast_lon`.

    Parameters
    ----------
    u : np.ndarray
        An array of eastward velocities.
    v : np.ndarray
        An array of northward velocities.
    time : np.ndarray
        An array of time values.
    x_origin : float
        Origin x-coordinate or origin longitude.
    y_origin : float
        Origin y-coordinate or origin latitude.
    coord_system : str, optional
        The coordinate system of the input. Can be "spherical" or "cartesian".
        Default is "spherical".
    integration_scheme : str, optional
        The difference scheme to use for computing the position. Can be
        "forward" or "backward". Default is "forward".
    time_axis : int, optional
        The axis of the time array. Default is -1, which corresponds to the
        last axis.

    Returns
    -------
    x : np.ndarray
        An array of zonal displacements or longitudes.
    y : np.ndarray
        An array of meridional displacements or latitudes.

    Examples
    --------

    Simple integration on a plane, using the forward scheme by default:

    >>> import numpy as np
    >>> from clouddrift.analysis import position_from_velocity
    >>> u = np.array([1., 2., 3., 4.])
    >>> v = np.array([1., 1., 1., 1.])
    >>> time = np.array([0., 1., 2., 3.])
    >>> x, y = position_from_velocity(u, v, time, 0, 0, coord_system="cartesian")
    >>> x
    array([0., 1., 3., 6.])
    >>> y
    array([0., 1., 2., 3.])

    As above, but using centered scheme:

    >>> x, y = position_from_velocity(u, v, time, 0, 0, coord_system="cartesian", integration_scheme="centered")
    >>> x
    array([0., 1.5, 4., 7.5])
    >>> y
    array([0., 1., 2., 3.])

    Simple integration on a sphere (default):

    >>> u = np.array([1., 2., 3., 4.])
    >>> v = np.array([1., 1., 1., 1.])
    >>> time = np.array([0., 1., 2., 3.]) * 1e5
    >>> x, y = position_from_velocity(u, v, time, 0, 0)
    >>> x
    array([0.        , 0.89839411, 2.69584476, 5.39367518])
    >>> y
    array([0.        , 0.89828369, 1.79601515, 2.69201609])

    Integrating across the antimeridian (dateline) by default does not
    recast the resulting longitude:

    >>> u = np.array([1., 1.])
    >>> v = np.array([0., 0.])
    >>> time = np.array([0, 1e5])
    >>> x, y = position_from_velocity(u, v, time, 179.5, 0)
    >>> x
    array([179.5      , 180.3983205])
    >>> y
    array([0., 0.])

    Use the ``clouddrift.sphere.recast_lon`` function to recast the longitudes
    to the desired range:

    >>> from clouddrift.sphere import recast_lon
    >>> recast_lon(x, -180)
    array([ 179.5      , -179.6016795])

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
        If the time axis is outside of the valid range ([-1, N-1]).
        If the input coordinate system is not "spherical" or "cartesian".
        If the input integration scheme is not "forward", "backward", or "centered"

    See Also
    --------
    :func:`velocity_from_position`
    """
    # Positions and time arrays must have the same shape.
    if not u.shape == v.shape == time.shape:
        raise ValueError("u, v, and time must have the same shape.")

    # time_axis must be in valid range
    if time_axis < -1 or time_axis > len(u.shape) - 1:
        raise ValueError(
            f"time_axis ({time_axis}) is outside of the valid range ([-1,"
            f" {len(x.shape) - 1}])."
        )

    # Nominal order of axes on input, i.e. (0, 1, 2, ..., N-1)
    target_axes = list(range(len(u.shape)))

    # If time_axis is not the last one, transpose the inputs
    if time_axis != -1 and time_axis < len(u.shape) - 1:
        target_axes.append(target_axes.pop(target_axes.index(time_axis)))

    # Reshape the inputs to ensure the time axis is last (fast-varying)
    u_ = np.transpose(u, target_axes)
    v_ = np.transpose(v, target_axes)
    time_ = np.transpose(time, target_axes)

    x = np.zeros(u_.shape, dtype=u.dtype)
    y = np.zeros(v_.shape, dtype=v.dtype)

    dt = np.diff(time_)

    if integration_scheme.lower() == "forward":
        x[..., 1:] = np.cumsum(u_[..., :-1] * dt, axis=-1)
        y[..., 1:] = np.cumsum(v_[..., :-1] * dt, axis=-1)
    elif integration_scheme.lower() == "backward":
        x[..., 1:] = np.cumsum(u_[1:] * dt, axis=-1)
        y[..., 1:] = np.cumsum(v_[1:] * dt, axis=-1)
    elif integration_scheme.lower() == "centered":
        x[..., 1:] = np.cumsum(0.5 * (u_[..., :-1] + u_[..., 1:]) * dt, axis=-1)
        y[..., 1:] = np.cumsum(0.5 * (v_[..., :-1] + v_[..., 1:]) * dt, axis=-1)
    else:
        raise ValueError(
            'integration_scheme must be "forward", "backward", or "centered".'
        )

    if coord_system.lower() == "cartesian":
        x += x_origin
        y += y_origin
    elif coord_system.lower() == "spherical":
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        bearings = np.arctan2(dy, dx)
        x[..., 0], y[..., 0] = x_origin, y_origin
        for n in range(distances.shape[-1]):
            y[..., n + 1], x[..., n + 1] = position_from_distance_and_bearing(
                y[..., n], x[..., n], distances[..., n], bearings[..., n]
            )
    else:
        raise ValueError('coord_system must be "spherical" or "cartesian".')

    if target_axes == list(range(len(u.shape))):
        return x, y
    else:
        return np.transpose(x, target_axes), np.transpose(y, target_axes)


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
    if zonal and meridional displacements are in the units of kilometers and
    time is in the units of hours, the resulting velocity is in the units of
    kilometers per hour.

    x, y, and time can be multi-dimensional arrays. If the time axis, along
    which the finite differencing is performed, is not the last one (i.e.
    x.shape[-1]), use the time_axis optional argument to specify along which
    axis should the differencing be done. x, y, and time must have the same
    shape.

    Difference scheme can take one of three values:

    #. "forward" (default): finite difference is evaluated as ``dx[i] = dx[i+1] - dx[i]``;
    #. "backward": finite difference is evaluated as ``dx[i] = dx[i] - dx[i-1]``;
    #. "centered": finite difference is evaluated as ``dx[i] = (dx[i+1] - dx[i-1]) / 2``.

    Forward and backward schemes are effectively the same except that the
    position at which the velocity is evaluated is shifted one element down in
    the backward scheme relative to the forward scheme. In the case of a
    forward or backward difference scheme, the last or first element of the
    velocity, respectively, is extrapolated from its neighboring point. In the
    case of a centered difference scheme, the start and end boundary points are
    evaluated using the forward and backward difference scheme, respectively.

    Parameters
    ----------
    x : array_like
        An N-d array of x-positions (longitude in degrees or zonal displacement in any unit)
    y : array_like
        An N-d array of y-positions (latitude in degrees or meridional displacement in any unit)
    time : array_like
        An N-d array of times as floating point values (in any unit)
    coord_system : str, optional
        Coordinate system that x and y arrays are in; possible values are "spherical" (default) or "cartesian".
    difference_scheme : str, optional
        Difference scheme to use; possible values are "forward", "backward", and "centered".
    time_axis : int, optional)
        Axis along which to differentiate (default is -1)

    Returns
    -------
    u : np.ndarray
        Zonal velocity
    v : np.ndarray
        Meridional velocity

    Raises
    ------
    ValueError
        If x, y, and time do not have the same shape.
        If time_axis is outside of the valid range.
        If coord_system is not "spherical" or "cartesian".
        If difference_scheme is not "forward", "backward", or "centered".

    See Also
    --------
    :func:`position_from_velocity`
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
            y1 = (y_[..., :-2] + y_[..., 1:-1]) / 2
            x1 = (x_[..., :-2] + x_[..., 1:-1]) / 2
            y2 = (y_[..., 2:] + y_[..., 1:-1]) / 2
            x2 = (x_[..., 2:] + x_[..., 1:-1]) / 2
            distances = distance(y1, x1, y2, x2)
            bearings = bearing(y1, x1, y2, x2)
            dx[..., 1:-1] = distances * np.cos(bearings)
            dy[..., 1:-1] = distances * np.sin(bearings)

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
    Criteria are combined on any data or metadata variables part of the Dataset. The following examples are based on the GDP dataset.

    Retrieve a region, like the Gulf of Mexico, using ranges of latitude and longitude:
    >>> subset(ds, {"lat": (21, 31), "lon": (-98, -78)})

    Retrieve drogued trajectory segments:
    >>> subset(ds, {"drogue_status": True})

    Retrieve trajectory segments with temperature higher than 25°C (303.15K):
    >>> subset(ds, {"sst": (303.15, np.inf)})

    Retrieve specific drifters from their IDs:
    >>> subset(ds, {"ID": [2578, 2582, 2583]})

    Retrieve a specific time period:
    >>> subset(ds, {"time": (np.datetime64("2000-01-01"), np.datetime64("2020-01-31"))})

    Note: To subset time variable, the range has to be defined as a function type of the variable. By default, `xarray` uses `np.datetime64` to represent datetime data. If the datetime data is a `datetime.datetime`, or `pd.Timestamp`, the range would have to be define accordingly.

    Those criteria can also be combined:
    >>> subset(ds, {"lat": (21, 31), "lon": (-98, -78), "drogue_status": True, "sst": (303.15, np.inf), "time": (np.datetime64("2000-01-01"), np.datetime64("2020-01-31"))})

    Raises
    ------
    ValueError
        If one of the variable in a criterion is not found in the Dataset
    """
    # Normally we expect the ragged-array dataset to have a "count" variable.
    # However, some datasets may have a "rowsize" variable instead, e.g. if they
    # have not gotten up to speed with our new convention. We check for both.
    if "count" in ds.variables:
        count_var = "count"
    elif "rowsize" in ds.variables:
        count_var = "rowsize"
    else:
        raise ValueError(
            "Ragged-array Dataset ds must have a 'count' or 'rowsize' variable."
        )

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
    traj_idx = np.insert(np.cumsum(ds[count_var].values), 0, 0)
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
        # apply the filtering for both dimensions
        ds_sub = ds.isel({"traj": mask_traj, "obs": mask_obs})
        # update the count
        ds_sub[count_var].values = segment(
            ds_sub.ids, 0.5, count=segment(ds_sub.ids, -0.5)
        )
        return ds_sub


def unpack_ragged(ragged_array: np.ndarray, count: np.ndarray[int]) -> list[np.ndarray]:
    """Unpack a ragged array into a list of regular arrays.

    Unpacking a ``np.ndarray`` ragged array is about 2 orders of magnitude
    faster than unpacking an ``xr.DataArray`` ragged array, so unless you need a
    ``DataArray`` as the result, we recommend passing ``np.ndarray`` as input.

    Parameters
    ----------
    ragged_array : array-like
        A ragged_array to unpack
    count : array-like
        An array of integers whose values is the size of each row in the ragged
        array

    Returns
    -------
    list
        A list of array-likes with sizes that correspond to the values in
        count, and types that correspond to the type of ragged_array

    Examples
    --------

    Unpacking longitude arrays from a ragged Xarray Dataset:

    .. code-block:: python

        lon = unpack_ragged(ds.lon, ds["count"]) # return a list[xr.DataArray] (slower)
        lon = unpack_ragged(ds.lon.values, ds["count"]) # return a list[np.ndarray] (faster)

    Looping over trajectories in a ragged Xarray Dataset to compute velocities
    for each:

    .. code-block:: python

        for lon, lat, time in list(zip(
            unpack_ragged(ds.lon.values, ds["count"]),
            unpack_ragged(ds.lat.values, ds["count"]),
            unpack_ragged(ds.time.values, ds["count"])
        )):
            u, v = velocity_from_position(lon, lat, time)
    """
    indices = np.insert(np.cumsum(np.array(count)), 0, 0)
    return [ragged_array[indices[n] : indices[n + 1]] for n in range(indices.size - 1)]

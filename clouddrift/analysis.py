import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
from concurrent import futures
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
    >>> def func(x, y):
    ...     return x + y
    >>> x = np.arange(10)
    >>> y = np.arange(10, 20)
    >>> apply_ragged(func, [x, y], [5, 5])
    array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

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


def segment(
    x: np.ndarray, tolerance: float, rowsize: np.ndarray[int] = None
) -> np.ndarray[int]:
    """Segment an array into contiguous segments.

    Parameters
    ----------
    x : list, np.ndarray, or xr.DataArray
        An array to segment.
    tolerance : float
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

    if rowsize is None:
        if tolerance >= 0:
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
            f"time_axis ({time_axis}) is outside of the valid range ([-1, {len(x.shape) - 1}])."
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

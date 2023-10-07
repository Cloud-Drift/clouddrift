"""
Transformational and inquiry functions for ragged arrays.
"""

import numpy as np
from typing import Tuple, Union, Iterable
import xarray as xr
import pandas as pd
from concurrent import futures
from datetime import timedelta
import warnings


def apply_ragged(
    func: callable,
    arrays: Union[list[Union[np.ndarray, xr.DataArray]], np.ndarray, xr.DataArray],
    rowsize: Union[list[int], np.ndarray[int], xr.DataArray],
    *args: tuple,
    rows: Union[int, Iterable[int]] = None,
    axis: int = 0,
    executor: futures.Executor = futures.ThreadPoolExecutor(max_workers=None),
    **kwargs: dict,
) -> Union[tuple[np.ndarray], np.ndarray]:
    """Apply a function to a ragged array.

    The function ``func`` will be applied to each contiguous row of ``arrays`` as
    indicated by row sizes ``rowsize``. The output of ``func`` will be
    concatenated into a single ragged array.

    You can pass ``arrays`` as NumPy arrays or xarray DataArrays, however,
    the result will always be a NumPy array. Passing ``rows`` as an integer or
    a sequence of integers will make ``apply_ragged`` process and return only
    those specific rows, and otherwise, all rows in the input ragged array will
    be processed. Further, you can use the ``axis`` parameter to specify the
    ragged axis of the input array(s) (default is 0).

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
    arrays : list[np.ndarray] or np.ndarray or xr.DataArray
        An array or a list of arrays to apply ``func`` to.
    rowsize : list[int] or np.ndarray[int] or xr.DataArray[int]
        List of integers specifying the number of data points in each row.
    *args : tuple
        Additional arguments to pass to ``func``.
    rows : int or Iterable[int], optional
        The row(s) of the ragged array to apply ``func`` to. If ``rows`` is
        ``None`` (default), then ``func`` will be applied to all rows.
    axis : int, optional
        The ragged axis of the input arrays. Default is 0.
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

    >>> rowsize = [2, 3, 4]
    >>> x = np.array([1, 2, 10, 12, 14, 30, 33, 36, 39])
    >>> y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> t = np.array([1, 2, 1, 2, 3, 1, 2, 3, 4])
    >>> u1, v1 = apply_ragged(velocity_from_position, [x, y, t], rowsize, coord_system="cartesian")
    array([1., 1., 2., 2., 2., 3., 3., 3., 3.]),
    array([1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    To apply ``func`` to only a subset of rows, use the ``rows`` argument:

    >>> u1, v1 = apply_ragged(velocity_from_position, [x, y, t], rowsize, rows=0, coord_system="cartesian")
    array([1., 1.]),
    array([1., 1.]))
    >>> u1, v1 = apply_ragged(velocity_from_position, [x, y, t], rowsize, rows=[0, 1], coord_system="cartesian")
    array([1., 1., 2., 2., 2.]),
    array([1., 1., 1., 1., 1.]))

    Raises
    ------
    ValueError
        If the sum of ``rowsize`` does not equal the length of ``arrays``.
    IndexError
        If empty ``arrays``.
    """
    # make sure the arrays is iterable
    if type(arrays) not in [list, tuple]:
        arrays = [arrays]
    # validate rowsize
    for arr in arrays:
        if not sum(rowsize) == arr.shape[axis]:
            raise ValueError("The sum of rowsize must equal the length of arr.")

    # split the array(s) into trajectories
    arrays = [unpack(np.array(arr), rowsize, rows, axis) for arr in arrays]
    iter = [[arrays[i][j] for i in range(len(arrays))] for j in range(len(arrays[0]))]

    # parallel execution
    res = [executor.submit(func, *x, *args, **kwargs) for x in iter]
    res = [r.result() for r in res]

    # Concatenate the outputs.

    # The following wraps items in a list if they are not already iterable.
    res = [item if isinstance(item, Iterable) else [item] for item in res]

    # np.concatenate can concatenate along non-zero axis iff the length of
    # arrays to be concatenated is > 1. If the length is 1, for example in the
    # case of func that reduces over the non-ragged axis, we can only
    # concatenate along axis 0.
    if isinstance(res[0], tuple):  # more than 1 parameter
        outputs = []
        for i in range(len(res[0])):  # iterate over each result variable
            # If we have multiple outputs and func is a reduction function,
            # we now here have a list of scalars. We need to wrap them in a
            # list to concatenate them.
            result = [r[i] if isinstance(r[i], Iterable) else [r[i]] for r in res]
            if len(result[0]) > 1:
                # Arrays to concatenate are longer than 1 element, so we can
                # concatenate along the non-zero axis.
                outputs.append(np.concatenate(result, axis=axis))
            else:
                # Arrays to concatenate are 1 element long, so we can only
                # concatenate along axis 0.
                outputs.append(np.concatenate(result))
        return tuple(outputs)
    else:
        if len(res[0]) > 1:
            # Arrays to concatenate are longer than 1 element, so we can
            # concatenate along the non-zero axis.
            return np.concatenate(res, axis=axis)
        else:
            # Arrays to concatenate are 1 element long, so we can only
            # concatenate along axis 0.
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

    ``chunk`` can be combined with :func:`apply_ragged` to chunk a ragged array.

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
    >>> rowsize = [2, 1, 2]
    >>> apply_ragged(chunk, x, rowsize, 2)
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
    rowsize: Union[list, np.ndarray, pd.Series, xr.DataArray],
    min_rowsize: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Within a ragged array, removes arrays less than a specified row size.

    Parameters
    ----------
    ragged : np.ndarray or pd.Series or xr.DataArray
        A ragged array.
    rowsize : list or np.ndarray[int] or pd.Series or xr.DataArray[int]
        The size of each row in the input ragged array.
    min_rowsize :
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
        If the sum of ``rowsize`` does not equal the length of ``arrays``.
    IndexError
        If empty ``ragged``.

    See Also
    --------
    :func:`segment`, `chunk`
    """

    ragged = apply_ragged(
        lambda x, min_len: x if len(x) >= min_len else np.empty(0, dtype=x.dtype),
        np.array(ragged),
        rowsize,
        min_len=min_rowsize,
    )
    rowsize = apply_ragged(
        lambda x, min_len: x if x >= min_len else np.empty(0, dtype=x.dtype),
        np.array(rowsize),
        np.ones_like(rowsize),
        min_len=min_rowsize,
    )

    return ragged, rowsize


def ragged_to_regular(
    ragged: Union[np.ndarray, pd.Series, xr.DataArray],
    rowsize: Union[list, np.ndarray, pd.Series, xr.DataArray],
    fill_value: float = np.nan,
) -> np.ndarray:
    """Convert a ragged array to a two-dimensional array such that each contiguous segment
    of a ragged array is a row in the two-dimensional array. Each row of the two-dimensional
    array is padded with NaNs as needed. The length of the first dimension of the output
    array is the length of ``rowsize``. The length of the second dimension is the maximum
    element of ``rowsize``.

    Note: Although this function accepts parameters of type ``xarray.DataArray``,
    passing NumPy arrays is recommended for performance reasons.

    Parameters
    ----------
    ragged : np.ndarray or pd.Series or xr.DataArray
        A ragged array.
    rowsize : list or np.ndarray[int] or pd.Series or xr.DataArray[int]
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
    res = fill_value * np.ones((len(rowsize), int(max(rowsize))), dtype=ragged.dtype)
    unpacked = unpack(ragged, rowsize)
    for n in range(len(rowsize)):
        res[n, : int(rowsize[n])] = unpacked[n]
    return res


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


def rowsize_to_index(rowsize: Union[list, np.ndarray, xr.DataArray]) -> np.ndarray:
    """Convert a list of row sizes to a list of indices.

    This function is typically used to obtain the indices of data rows organized
    in a ragged array.

    Parameters
    ----------
    rowsize : list or np.ndarray or xr.DataArray
        A list of row sizes.

    Returns
    -------
    np.ndarray
        A list of indices.

    Examples
    --------

    To obtain the indices within a ragged array of three consecutive rows of sizes 100, 202, and 53:

    >>> rowsize_to_index([100, 202, 53])
    array([0, 100, 302, 355])
    """
    return np.cumsum(np.insert(np.array(rowsize), 0, 0))


def segment(
    x: np.ndarray,
    tolerance: Union[float, np.timedelta64, timedelta, pd.Timedelta],
    rowsize: np.ndarray[int] = None,
) -> np.ndarray[int]:
    """Divide an array into segments based on a tolerance value.

    Parameters
    ----------
    x : list, np.ndarray, or xr.DataArray
        An array to divide into segment.
    tolerance : float, np.timedelta64, timedelta, pd.Timedelta
        The maximum signed difference between consecutive points in a segment.
        The array x will be segmented wherever differences exceed the tolerance.
    rowsize : np.ndarray[int], optional
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
    a ragged array), then the ``rowsize`` argument can be used to preserve
    the original segments:

    >>> x = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
    >>> rowsize = [3, 2, 6]
    >>> segment(x, 0.5, rowsize)
    array([1, 2, 1, 1, 1, 4, 1])

    The tolerance can also be negative. In this case, the input array is
    segmented where the negative difference exceeds the negative
    value of the tolerance, i.e. where ``x[n+1] - x[n] < -tolerance``:

    >>> x = [0, 1, 2, 0, 1, 2]
    >>> segment(x, -0.5)
    array([3, 3])

    To segment an array for both positive and negative gaps, invoke the function
    twice, once for a positive tolerance and once for a negative tolerance.
    The result of the first invocation can be passed as the ``rowsize`` argument
    to the first ``segment`` invocation:

    >>> x = [1, 1, 2, 2, 1, 1, 2, 2]
    >>> segment(x, 0.5, rowsize=segment(x, -0.5))
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


def subset(
    ds: xr.Dataset,
    criteria: dict,
    id_var_name: str = "ID",
    rowsize_var_name: str = "rowsize",
    traj_dim_name: str = "traj",
    obs_dim_name: str = "obs",
) -> xr.Dataset:
    """Subset the dataset as a function of one or many criteria. The criteria are
    passed as a dictionary, where a variable to subset is assigned to either a
    range (valuemin, valuemax), a list [value1, value2, valueN], or a single value.

    This function relies on specific names of the dataset dimensions and the
    rowsize variables. The default expected values are listed in the Parameters
    section, however, if your dataset uses different names for these dimensions
    and variables, you can specify them using the optional arguments.

    Parameters
    ----------
    ds : xr.Dataset
        Lagrangian dataset stored in two-dimensional or ragged array format
    criteria : dict
        dictionary containing the variables and the ranges/values to subset
    id_var_name : str, optional
        Name of the variable containing the ID of the trajectories (default is "ID")
    rowsize_var_name : str, optional
        Name of the variable containing the number of observations per trajectory (default is "rowsize")
    traj_dim_name : str, optional
        Name of the trajectory dimension (default is "traj")
    obs_dim_name : str, optional
        Name of the observation dimension (default is "obs")

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

    Note that to subset time variable, the range has to be defined as a function
    type of the variable. By default, ``xarray`` uses ``np.datetime64`` to
    represent datetime data. If the datetime data is a ``datetime.datetime``, or
    ``pd.Timestamp``, the range would have to be defined accordingly.

    Those criteria can also be combined:
    >>> subset(ds, {"lat": (21, 31), "lon": (-98, -78), "drogue_status": True, "sst": (303.15, np.inf), "time": (np.datetime64("2000-01-01"), np.datetime64("2020-01-31"))})

    Raises
    ------
    ValueError
        If one of the variable in a criterion is not found in the Dataset
    """
    mask_traj = xr.DataArray(
        data=np.ones(ds.dims[traj_dim_name], dtype="bool"), dims=[traj_dim_name]
    )
    mask_obs = xr.DataArray(
        data=np.ones(ds.dims[obs_dim_name], dtype="bool"), dims=[obs_dim_name]
    )

    for key in criteria.keys():
        if key in ds:
            if ds[key].dims == (traj_dim_name,):
                mask_traj = np.logical_and(mask_traj, _mask_var(ds[key], criteria[key]))
            elif ds[key].dims == (obs_dim_name,):
                mask_obs = np.logical_and(mask_obs, _mask_var(ds[key], criteria[key]))
        else:
            raise ValueError(f"Unknown variable '{key}'.")

    # remove data when trajectories are filtered
    traj_idx = rowsize_to_index(ds[rowsize_var_name].values)
    for i in np.where(~mask_traj)[0]:
        mask_obs[slice(traj_idx[i], traj_idx[i + 1])] = False

    # remove trajectory completely filtered in mask_obs
    ids_with_mask_obs = np.repeat(ds[id_var_name].values, ds[rowsize_var_name].values)[
        mask_obs
    ]
    mask_traj = np.logical_and(
        mask_traj, np.in1d(ds[id_var_name], np.unique(ids_with_mask_obs))
    )

    if not any(mask_traj):
        warnings.warn("No data matches the criteria; returning an empty dataset.")
        return xr.Dataset()
    else:
        # apply the filtering for both dimensions
        ds_sub = ds.isel({traj_dim_name: mask_traj, obs_dim_name: mask_obs})
        _, unique_idx, sorted_rowsize = np.unique(
            ids_with_mask_obs, return_index=True, return_counts=True
        )
        ds_sub[rowsize_var_name].values = sorted_rowsize[np.argsort(unique_idx)]
        return ds_sub


def unpack(
    ragged_array: np.ndarray,
    rowsize: np.ndarray[int],
    rows: Union[int, Iterable[int]] = None,
    axis: int = 0,
) -> list[np.ndarray]:
    """Unpack a ragged array into a list of regular arrays.

    Unpacking a ``np.ndarray`` ragged array is about 2 orders of magnitude
    faster than unpacking an ``xr.DataArray`` ragged array, so unless you need a
    ``DataArray`` as the result, we recommend passing ``np.ndarray`` as input.

    Parameters
    ----------
    ragged_array : array-like
        A ragged_array to unpack
    rowsize : array-like
        An array of integers whose values is the size of each row in the ragged
        array
    rows : int or Iterable[int], optional
        A row or list of rows to unpack. Default is None, which unpacks all rows.
    axis : int, optional
        The axis along which to unpack the ragged array. Default is 0.

    Returns
    -------
    list
        A list of array-likes with sizes that correspond to the values in
        rowsize, and types that correspond to the type of ragged_array

    Examples
    --------

    Unpacking longitude arrays from a ragged Xarray Dataset:

    .. code-block:: python

        lon = unpack(ds.lon, ds["rowsize"]) # return a list[xr.DataArray] (slower)
        lon = unpack(ds.lon.values, ds["rowsize"]) # return a list[np.ndarray] (faster)
        first_lon = unpack(ds.lon.values, ds["rowsize"], rows=0) # return only the first row
        first_two_lons = unpack(ds.lon.values, ds["rowsize"], rows=[0, 1]) # return first two rows

    Looping over trajectories in a ragged Xarray Dataset to compute velocities
    for each:

    .. code-block:: python

        for lon, lat, time in list(zip(
            unpack(ds.lon.values, ds["rowsize"]),
            unpack(ds.lat.values, ds["rowsize"]),
            unpack(ds.time.values, ds["rowsize"])
        )):
            u, v = velocity_from_position(lon, lat, time)
    """
    indices = rowsize_to_index(rowsize)

    if rows is None:
        rows = range(indices.size - 1)
    if isinstance(rows, int):
        rows = [rows]

    unpacked = np.split(ragged_array, indices[1:-1], axis=axis)

    return [unpacked[i] for i in rows]


def _mask_var(
    var: xr.DataArray,
    criterion: Union[tuple, list, np.ndarray, xr.DataArray, bool, float, int],
) -> xr.DataArray:
    """Return the mask of a subset of the data matching a test criterion.

    Parameters
    ----------
    var : xr.DataArray
        DataArray to be subset by the criterion
    criterion : array-like
        The criterion can take three forms:
        - tuple: (min, max) defining a range
        - list, np.ndarray, or xr.DataArray: An array-like defining multiples values
        - scalar: value defining a single value

    Examples
    --------
    >>> x = xr.DataArray(data=np.arange(0, 5))
    >>> _mask_var(x, (2, 4))
    <xarray.DataArray (dim_0: 5)>
    array([False, False,  True,  True,  True])
    Dimensions without coordinates: dim_0

    >>> _mask_var(x, [0, 2, 4])
    <xarray.DataArray (dim_0: 5)>
    array([ True, False, True,  False, True])
    Dimensions without coordinates: dim_0

    >>> _mask_var(x, 4)
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
    elif isinstance(
        criterion, (list, np.ndarray, xr.DataArray)
    ):  # select multiple values
        # Ensure we define the mask as boolean, otherwise it will inherit
        # the dtype of the variable which may be a string, object, or other.
        mask = xr.zeros_like(var, dtype=bool)
        for v in criterion:
            mask = np.logical_or(mask, var == v)
    else:  # select one specific value
        mask = var == criterion
    return mask

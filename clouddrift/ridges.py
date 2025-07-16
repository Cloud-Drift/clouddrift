from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import integrate


def gradient_of_angle(
    x: NDArray[np.float64 | np.float64 | np.float32 | np.float64],
    edge_order: int = 1,
    axis: int = -1,
    discont: float = np.pi,
) -> NDArray[np.float64 | np.complex128 | np.float32 | np.complex64]:
    """
    Compute the gradient of an angle array with proper unwrapping.

    This function is a specialized version of np.gradient that handles phase angle unwrapping
    to avoid discontinuities at ±π. It is designed for use with arrays of angles in radians.

    Parameters
    ----------
    x : np.ndarray
        Array of angles (in radians) to compute the gradient of.
    edge_order : int, optional
        How the edges are handled during the gradient computation. See np.gradient (default: 1).
    axis : int, optional
        Axis along which the gradient is computed (default: -1).
    discont : float, optional
        Discontinuity parameter for np.unwrap (default: np.pi).

    Returns
    -------
    result : np.ndarray
        The gradient of the phase-angle array, with the same shape as `x`.

    Notes
    -----
    This function was originally written at:
    https://github.com/danrsc/analytic_wavelet
    """
    # based on the np.gradient code
    slice_interior_source_even = [slice(None)] * x.ndim
    slice_interior_source_odd = [slice(None)] * x.ndim
    slice_interior_dest_of_source_even = [slice(None)] * x.ndim
    slice_interior_dest_of_source_odd = [slice(None)] * x.ndim
    slice_begin_source = [slice(None)] * x.ndim
    slice_end_source = [slice(None)] * x.ndim
    slice_begin_dest = [slice(None)] * x.ndim
    slice_end_dest = [slice(None)] * x.ndim

    is_even = x.shape[axis] // 2 * 2 == x.shape[axis]

    slice_interior_source_even[axis] = slice(None, None, 2)
    slice_interior_source_odd[axis] = slice(1, None, 2)
    if is_even:
        slice_interior_dest_of_source_even[axis] = slice(1, -1, 2)
        slice_interior_dest_of_source_odd[axis] = slice(2, None, 2)
    else:
        slice_interior_dest_of_source_even[axis] = slice(1, None, 2)
        slice_interior_dest_of_source_odd[axis] = slice(2, -1, 2)

    result = np.empty_like(x)

    result[tuple(slice_interior_dest_of_source_even)] = (
        np.diff(
            np.unwrap(x[tuple(slice_interior_source_even)], axis=axis, discont=discont),
            axis=axis,
        )
        / 2
    )
    result[tuple(slice_interior_dest_of_source_odd)] = (
        np.diff(
            np.unwrap(x[tuple(slice_interior_source_odd)], axis=axis, discont=discont),
            axis=axis,
        )
        / 2
    )

    slice_begin_dest[axis] = slice(0, 1)
    slice_end_dest[axis] = slice(-1, None)
    if edge_order == 1:
        slice_begin_source[axis] = slice(0, 2)
        slice_end_source[axis] = slice(-2, None)
        result[tuple(slice_begin_dest)] = np.diff(
            np.unwrap(x[tuple(slice_begin_source)], axis=axis, discont=discont),
            axis=axis,
        )
        result[tuple(slice_end_dest)] = np.diff(
            np.unwrap(x[tuple(slice_end_source)], axis=axis, discont=discont), axis=axis
        )
    elif edge_order == 2:
        slice_begin_source[axis] = slice(1, 2)
        slice_end_source[axis] = slice(-2, -1)
        begin_source = np.concatenate(
            [x[tuple(slice_end_dest)], x[tuple(slice_begin_source)]], axis=axis
        )
        end_source = np.concatenate(
            [x[tuple(slice_end_source)], x[tuple(slice_begin_dest)]], axis=axis
        )
        result[tuple(slice_begin_dest)] = (
            np.diff(np.unwrap(begin_source, axis=axis, discont=discont), axis=axis) / 2
        )
        result[tuple(slice_end_dest)] = (
            np.diff(np.unwrap(end_source, axis=axis, discont=discont), axis=axis) / 2
        )
    else:
        raise ValueError("Unexpected edge_order: {}".format(edge_order))

    return result


def instmom_univariate(
    signal: NDArray[np.complex128], sample_rate: float = 1.0, axis: int = -1
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    """
    Calculate univariate instantaneous moments for a single signal.

    This function computes the instantaneous amplitude, frequency, bandwidth, and curvature
    for a single signal.

    Parameters
    ----------
    signal : NDArray[np.complex128]
        Input signal array (complex-valued) (shape: (scale, time))
    sample_rate : float, optional
        Sample rate for time derivatives, defaults to 1.0
    axis : int, optional
        Axis representing time, defaults to 0

    Returns
    -------
    amplitude : NDArray[np.float64]
        Instantaneous amplitude
    omega : NDArray[np.float64]
        Instantaneous radian frequency
    upsilon : NDArray[np.float64]
        Instantaneous bandwidth
    xi : NDArray[np.complex128]
        Instantaneous curvature
    """
    # Calculate amplitude
    amplitude = np.abs(signal).astype(np.float64)

    # Calculate instantaneous frequency
    angle_data = np.angle(signal).astype(np.float64)
    omega = (gradient_of_angle(angle_data, axis=axis) * sample_rate).astype(np.float64)

    # Calculate instantaneous bandwidth
    log_amplitude = np.log(amplitude)
    upsilon = (np.gradient(log_amplitude, axis=axis) * sample_rate).astype(np.float64)

    # Calculate instantaneous curvature
    d_omega = (np.gradient(omega, axis=axis) * sample_rate).astype(np.float64)
    d_upsilon = (np.gradient(upsilon, axis=axis) * sample_rate).astype(np.float64)
    xi = (upsilon**2 + d_upsilon + 1j * d_omega).astype(np.complex128)

    return amplitude, omega, upsilon, xi


def instmom_multivariate(
    signals: NDArray[np.complex128],
    sample_rate: float = 1.0,
    time_axis: int = -1,
    joint_axis: int = -1,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Calculate joint instantaneous moments across multiple signals.

    This function computes the joint amplitude, frequency, bandwidth, and curvature
    across multiple signals, with power-weighted averaging of component properties.

    Parameters
    ----------
    signals : NDArray[np.complex128]
        Input array with shape (..., n_signals) where n_signals is the number
        of signals across which to compute joint moments. (shape: (scale, time, n_signals))
    sample_rate : float, optional
        Sample rate for time derivatives, defaults to 1.0
    time_axis : int, optional
        Axis representing time, defaults to 0
    joint_axis : int, optional
        Axis across which to compute joint moments, defaults to -1

    Returns
    -------
    joint_amplitude : NDArray[np.float64]
        Joint instantaneous amplitude (root-mean-square across signals)
    joint_omega : NDArray[np.float64]
        Joint instantaneous radian frequency (power-weighted average)
    joint_upsilon : NDArray[np.float64]
        Joint instantaneous bandwidth
    joint_xi : NDArray[np.float64]
        Joint instantaneous curvature

    Notes
    -----
    Reference: Lilly & Olhede (2010)
    """
    # Calculate univariate moments for each signal
    amplitude = np.abs(signals)

    # Calculate instantaneous radian frequency for each signal
    phase = np.angle(signals)
    omega = gradient_of_angle(phase, axis=time_axis) * sample_rate

    # Calculate instantaneous bandwidth for each signal
    log_amplitude = np.log(amplitude)
    upsilon = np.gradient(log_amplitude, axis=time_axis) * sample_rate

    # Calculate instantaneous curvature for each signal
    d_omega = np.gradient(omega, axis=time_axis) * sample_rate
    d_upsilon = np.gradient(upsilon, axis=time_axis) * sample_rate
    xi = upsilon**2 + d_upsilon + 1j * d_omega

    # Calculate joint amplitude (first moment)
    squared_amplitude = amplitude**2
    joint_amplitude = np.sqrt(
        np.mean(squared_amplitude, axis=joint_axis, keepdims=True)
    )

    # Calculate joint frequency (second moment)
    weights = squared_amplitude
    # Normalize weights along joint axis
    weights_sum = np.sum(weights, axis=joint_axis, keepdims=True)
    normalized_weights = weights / weights_sum

    joint_omega = np.sum(omega * normalized_weights, axis=joint_axis, keepdims=True)

    # Create replicated joint frequency array for deviation calculations
    # This creates a broadcast-compatible array with joint_omega expanded along joint_axis
    broadcast_shape = list(signals.shape)
    broadcast_shape[joint_axis] = 1
    omega_mean = np.broadcast_to(joint_omega, signals.shape)

    # Calculate joint bandwidth (third moment)
    deviation = upsilon + 1j * (omega - omega_mean)
    joint_upsilon = np.sqrt(
        np.sum(
            np.abs(deviation) ** 2 * normalized_weights, axis=joint_axis, keepdims=True
        )
    )

    # Calculate joint curvature (fourth moment)
    curvature_deviation = (
        xi + 2j * upsilon * (omega - omega_mean) - (omega - omega_mean) ** 2
    )
    joint_xi = np.sqrt(
        np.sum(
            np.abs(curvature_deviation) ** 2 * normalized_weights,
            axis=joint_axis,
            keepdims=True,
        )
    )

    # Remove singleton dimensions created by keepdims=True
    joint_amplitude = np.squeeze(joint_amplitude, axis=joint_axis)
    joint_omega = np.squeeze(joint_omega, axis=joint_axis)
    joint_upsilon = np.squeeze(joint_upsilon, axis=joint_axis)
    joint_xi = np.squeeze(joint_xi, axis=joint_axis)

    return joint_amplitude, joint_omega, joint_upsilon, joint_xi


def isridgepoint(
    wavelet_transform: NDArray[np.complex128],
    scale_frequencies: NDArray[np.float64],
    amplitude_threshold: float,
    ridge_type: str,
    freq_min: Optional[Union[float, NDArray[np.float64]]] = None,
    freq_max: Optional[Union[float, NDArray[np.float64]]] = None,
    mask: Optional[NDArray[np.bool_]] = None,
) -> Tuple[
    NDArray[np.bool_], NDArray[np.float64], NDArray[np.complex128], NDArray[np.float64]
]:
    """
    Find wavelet ridge points using specified criterion.
    Ridge detection is performed by finding local maxima along the time axis.

    Parameters
    ----------
    wavelet_transform : NDArray[np.complex128]
        Wavelet transform matrix with shape (time, scale)
    scale_frequencies : NDArray[np.float64]
        Frequencies corresponding to wavelet scales (in radians)
    amplitude_threshold : float
        Minimum amplitude threshold for ridge points
    ridge_type : str
        Ridge definition: 'amplitude' or 'phase'
    freq_min : float or NDArray[np.float64], optional
        Minimum frequency constraint
    freq_max : float or NDArray[np.float64], optional
        Maximum frequency constraint
    mask : NDArray[np.bool_], optional
        Boolean mask to restrict ridge locations

    Returns
    -------
    ridge_points : NDArray[np.bool_]
        Boolean matrix indicating ridge points
    ridge_quantity : NDArray[np.float64]
        Ridge quantity used for detection
    processed_transform : NDArray[np.complex128]
        Processed wavelet transform
    inst_frequency : NDArray[np.float64]
        Instantaneous frequency of the transform
    """
    # Handle empty arrays
    if wavelet_transform.size == 0:
        empty_shape = wavelet_transform.shape
        return (
            np.zeros(empty_shape, dtype=bool),
            np.zeros(empty_shape, dtype=np.float64),
            np.zeros(empty_shape, dtype=np.complex128),
            np.zeros(empty_shape, dtype=np.float64),
        )

    # Calculate instantaneous moments
    if wavelet_transform.ndim > 2 and wavelet_transform.shape[2] > 1:
        # Multivariate case
        amplitude, inst_frequency = instmom_multivariate(
            wavelet_transform, time_axis=-1, joint_axis=2
        )[:2]
    else:
        # Univariate case
        amplitude, inst_frequency = instmom_univariate(wavelet_transform, axis=-1)[:2]

    # Determine ridge quantity based on ridge type
    if ridge_type.lower().startswith("amp"):
        ridge_quantity = amplitude
    else:  # phase-based ridges
        # Create array of scale frequencies matching time dimension
        freq_matrix = np.broadcast_to(
            scale_frequencies[:, np.newaxis],
            wavelet_transform.shape,
        )
        ridge_quantity = inst_frequency - freq_matrix

    # Handle univariate and multivariate cases
    if wavelet_transform.ndim > 2 and wavelet_transform.shape[2] > 1:
        # Calculate magnitude
        transform_magnitude = np.abs(wavelet_transform)

        # Compute power-weighted phase average
        weight_sum = np.sum(transform_magnitude * wavelet_transform, axis=2)
        weight_total = np.sum(transform_magnitude**2, axis=2)

        # Safe division for weighted average
        phase_average = np.zeros_like(weight_sum, dtype=complex)
        mask_nonzero = weight_total != 0
        phase_average[mask_nonzero] = (
            weight_sum[mask_nonzero] / weight_total[mask_nonzero]
        )

        # Create joint transform with combined magnitude and average phase
        joint_magnitude = np.sqrt(np.sum(transform_magnitude**2, axis=2))
        joint_phase = np.angle(phase_average)
        processed_transform = joint_magnitude * np.exp(1j * joint_phase)
    else:
        processed_transform = wavelet_transform

    # Define ridge points array
    ridge_points = np.zeros(ridge_quantity.shape, dtype=bool)

    # Process all time points except edges
    if ridge_type.lower().startswith("amp"):
        # For amplitude ridges: find local maxima in time
        for i in range(1, ridge_quantity.shape[0] - 1):
            ridge_points[i, :] = (ridge_quantity[i - 1, :] < ridge_quantity[i, :]) & (
                ridge_quantity[i + 1, :] < ridge_quantity[i, :]
            )
    else:
        # For phase ridges: check freq_matrix monotonicity and find zero crossings
        freq_diff = np.diff(freq_matrix, axis=0)
        is_monotonic_increasing = np.all(freq_diff >= 0, axis=0)
        is_monotonic_decreasing = np.all(freq_diff <= 0, axis=0)
        is_monotonic = is_monotonic_increasing | is_monotonic_decreasing

        if not np.all(is_monotonic):
            non_monotonic_cols = np.where(~is_monotonic)[0]
            raise ValueError(
                f"freq_matrix must be monotonically increasing or decreasing along the scale axis. "
                f"Non-monotonic behavior detected at time indices: {non_monotonic_cols}"
            )

        # Determine if freq_matrix is generally increasing or decreasing
        freq_is_increasing = np.all(is_monotonic_increasing)

        for i in range(1, ridge_quantity.shape[0] - 1):
            if freq_is_increasing:
                # For increasing frequency: look for positive to negative crossings
                ridge_points[i, :] = (
                    (ridge_quantity[i - 1, :] > 0) & (ridge_quantity[i + 1, :] <= 0)
                ) | ((ridge_quantity[i - 1, :] >= 0) & (ridge_quantity[i + 1, :] < 0))
            else:
                # For decreasing frequency: look for negative to positive crossings
                ridge_points[i, :] = (
                    (ridge_quantity[i - 1, :] < 0) & (ridge_quantity[i + 1, :] >= 0)
                ) | ((ridge_quantity[i - 1, :] <= 0) & (ridge_quantity[i + 1, :] > 0))

    # Ensure we have the strongest local extrema
    error_matrix = np.abs(ridge_quantity)

    # Remove points where error is larger than adjacent time points
    for i in range(1, ridge_quantity.shape[0] - 1):
        # Check previous time step
        if i > 1:
            is_prev_ridge = ridge_points[i - 1, :]
            is_bigger_than_prev = error_matrix[i, :] > error_matrix[i - 1, :]
            ridge_points[i, :] = ridge_points[i, :] & ~(
                is_prev_ridge & is_bigger_than_prev
            )

        # Check next time step
        if i < ridge_quantity.shape[0] - 2:
            is_next_ridge = ridge_points[i + 1, :]
            is_bigger_than_next = error_matrix[i, :] > error_matrix[i + 1, :]
            ridge_points[i, :] = ridge_points[i, :] & ~(
                is_next_ridge & is_bigger_than_next
            )

    # Apply basic filtering criteria
    ridge_points = ridge_points & ~np.isnan(processed_transform)
    ridge_points = ridge_points & (np.abs(processed_transform) >= amplitude_threshold)

    # Apply frequency constraints if provided
    if freq_min is not None and freq_max is not None:
        # Create frequency constraint matrices
        if np.isscalar(freq_min) or (
            isinstance(freq_min, np.ndarray) and freq_min.size == 1
        ):
            freq_min_matrix = np.full(processed_transform.shape[:2], freq_min)
        else:
            # Ensure freq_min is an array before indexing
            freq_min_array = np.asarray(freq_min)
            # Expand row vector to match dimensions
            freq_min_matrix = np.broadcast_to(
                freq_min_array[:, np.newaxis], processed_transform.shape[:2]
            )

        if np.isscalar(freq_max) or (
            isinstance(freq_max, np.ndarray) and freq_max.size == 1
        ):
            freq_max_matrix = np.full(processed_transform.shape[:2], freq_max)
        else:
            # Ensure freq_max is an array before indexing
            freq_max_array = np.asarray(freq_max)
            # Expand row vector to match dimensions
            freq_max_matrix = np.broadcast_to(
                freq_max_array[:, np.newaxis], processed_transform.shape[:2]
            )

        # Apply frequency constraints
        freq_constraint = (inst_frequency > freq_min_matrix) & (
            inst_frequency < freq_max_matrix
        )
        ridge_points = ridge_points & freq_constraint

    # Apply additional mask if provided
    if mask is not None:
        ridge_points = ridge_points & mask

    return ridge_points, ridge_quantity, processed_transform, inst_frequency


def ridge_shift_interpolation(
    ridge_points: NDArray[np.bool_],
    ridge_quantity: NDArray[np.float64],
    y_arrays: List[NDArray[np.float64]],
) -> List[NDArray[np.float64]]:
    """
    Interpolates ridge quantities and arrays based on the maximum of a quadratic fit
    to the ridge quantity. This is done to improve the accuracy of the ridge along the
    frequency axis.

    Parameters
    ----------
    ridge_points : NDArray[np.bool_]
        Boolean mask indicating ridge points
    ridge_quantity : NDArray[np.float64]
        Ridge quantity used for interpolation
    y_arrays : List[NDArray[np.float64]]
        List of arrays to be interpolated at the new scale values.

    Returns
    -------
    y_values_interpolated : List[NDArray[np.float64]]

    """
    # Skip if no ridge points
    if not np.any(ridge_points):
        print("No ridge points found, returning empty result.")

        return [np.array([], dtype=y_array.dtype) for y_array in y_arrays]

    # Get indices of ridge points
    freq_indices, time_indices = np.where(ridge_points)

    # Initialize output value arrays (1D arrays for each y_array's ridge points)
    y_values_interpolated = [
        np.zeros(len(freq_indices), dtype=y_array.dtype) for y_array in y_arrays
    ]

    # Create masks for boundary points
    boundary_mask = (freq_indices <= 0) | (freq_indices >= ridge_quantity.shape[0] - 1)
    interior_mask = ~boundary_mask

    # Process boundary points (just copy original values)
    if np.any(boundary_mask):
        boundary_freq = freq_indices[boundary_mask]
        boundary_time = time_indices[boundary_mask]
        for j, y_array in enumerate(y_arrays):
            y_values_interpolated[j][boundary_mask] = y_array[
                boundary_freq, boundary_time
            ]

    # Process interior points using vectorized operations
    if np.any(interior_mask):
        int_freq = freq_indices[interior_mask]
        int_time = time_indices[interior_mask]

        # Get ridge quantities at points and neighbors
        ridge_curr = ridge_quantity[int_freq, int_time]
        ridge_prev = ridge_quantity[int_freq - 1, int_time]
        ridge_next = ridge_quantity[int_freq + 1, int_time]

        # Quadratic coefficients (vectorized)
        a = 0.5 * (ridge_next + ridge_prev - 2 * ridge_curr)
        b = 0.5 * (ridge_next - ridge_prev)

        # Find points with valid maximum (a < 0)
        max_mask = a < 0

        if np.any(max_mask):
            # Calculate x_max for points with valid maximum
            x_max = -b[max_mask] / (2 * a[max_mask])

            # Filter for points with x_max in range [-1, 1]
            valid_range_mask = (-1 <= x_max) & (x_max <= 1)

            if np.any(valid_range_mask):
                # Get indices for points with valid quadratic maximum
                quad_indices = np.where(max_mask)[0][valid_range_mask]
                quad_freq = int_freq[max_mask][valid_range_mask]
                quad_time = int_time[max_mask][valid_range_mask]
                quad_x_max = x_max[valid_range_mask]

                # Process each y_array using quadratic interpolation
                for j, y_array in enumerate(y_arrays):
                    y_prev = y_array[quad_freq - 1, quad_time]
                    y_curr = y_array[quad_freq, quad_time]
                    y_next = y_array[quad_freq + 1, quad_time]

                    # Quadratic coefficients for each array
                    y_a = 0.5 * (y_next + y_prev - 2 * y_curr)
                    y_b = 0.5 * (y_next - y_prev)
                    y_c = y_curr

                    # Evaluate quadratic at interpolated positions
                    y_interp = y_a * quad_x_max**2 + y_b * quad_x_max + y_c

                    # Store in our 1D array at the right indices (apply mask to interior points)
                    interior_indices = np.where(interior_mask)[0][quad_indices]
                    y_values_interpolated[j][interior_indices] = y_interp

        # Handle points that need linear interpolation
        linear_mask = np.ones(interior_mask.sum(), dtype=bool)
        if np.any(max_mask):
            linear_mask[max_mask] = ~valid_range_mask

        if np.any(linear_mask):
            # Get indices to update in the result arrays
            lin_indices = np.where(interior_mask)[0][linear_mask]
            lin_freq = int_freq[linear_mask]
            lin_time = int_time[linear_mask]

            # Get ridge values for linear points
            lin_curr = ridge_curr[linear_mask]
            lin_prev = ridge_prev[linear_mask]
            lin_next = ridge_next[linear_mask]

            # Determine direction for interpolation
            prev_higher = lin_prev >= lin_next

            # Calculate weights, limited to 0.5 for stability
            weights = np.zeros_like(lin_curr, dtype=np.float64)

            # For points where previous is higher
            if np.any(prev_higher):
                # Avoid division by zero
                denom = lin_prev[prev_higher] - lin_curr[prev_higher]
                nonzero = denom != 0
                if np.any(nonzero):
                    w = (
                        lin_prev[prev_higher][nonzero] - lin_curr[prev_higher][nonzero]
                    ) / denom[nonzero]
                    weights[prev_higher][nonzero] = np.minimum(0.5, w)

            # For points where next is higher
            next_higher = ~prev_higher
            if np.any(next_higher):
                # Avoid division by zero
                denom = lin_next[next_higher] - lin_curr[next_higher]
                nonzero = denom != 0
                if np.any(nonzero):
                    w = (
                        lin_next[next_higher][nonzero] - lin_curr[next_higher][nonzero]
                    ) / denom[nonzero]
                    weights[next_higher][nonzero] = np.minimum(0.5, w)

            # Apply linear interpolation for each y_array
            for j, y_array in enumerate(y_arrays):
                y_interp = np.zeros_like(lin_curr, dtype=y_array.dtype)

                # Points where previous is higher
                if np.any(prev_higher):
                    prev_weights = weights[prev_higher]
                    y_curr = y_array[lin_freq[prev_higher], lin_time[prev_higher]]
                    y_prev = y_array[lin_freq[prev_higher] - 1, lin_time[prev_higher]]
                    y_interp[prev_higher] = (
                        1 - prev_weights
                    ) * y_curr + prev_weights * y_prev

                # Points where next is higher
                if np.any(next_higher):
                    next_weights = weights[next_higher]
                    y_curr = y_array[lin_freq[next_higher], lin_time[next_higher]]
                    y_next = y_array[lin_freq[next_higher] + 1, lin_time[next_higher]]
                    y_interp[next_higher] = (
                        1 - next_weights
                    ) * y_curr + next_weights * y_next

                # Store the interpolated values in the result array
                y_values_interpolated[j][lin_indices] = y_interp

    # Return results in coordinate format
    return y_values_interpolated


def organize_ridge_points(
    freq_indices: NDArray[np.int_],
    time_indices: NDArray[np.int_],
    transform_shape: Tuple[int, int],
    arrays_to_organize: List[NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]]]:
    """
    Organizes ridge points and associated data into time-ridge matrices.

    Parameters
    ----------
    freq_indices : NDArray[np.int_]
        Array of frequency/scale indices for ridge points
    time_indices : NDArray[np.int_]
        Array of time indices for ridge points
    transform_shape : tuple
        Shape of the wavelet transform (scale, time)
    arrays_to_organize : List[NDArray[np.float64]]
        List of arrays containing data for each ridge point

    Returns
    -------
    index_matrix : NDArray[np.float64]
        Matrix of shape (time, max_ridges) mapping original indices to time-ridge coordinates
    organized_arrays : List[NDArray[np.float64]]
        List of matrices of shape (time, max_ridges) containing the organized data
    """
    # Sort ridge points by time
    sort_idx = np.lexsort((freq_indices, time_indices))
    freq_indices_sorted = freq_indices[sort_idx]
    time_indices_sorted = time_indices[sort_idx]

    # Create sorted versions of the input arrays
    sorted_arrays = [arr[sort_idx] for arr in arrays_to_organize]

    # Create a boolean matrix marking ridge point positions
    ridge_mask = np.zeros(transform_shape, dtype=bool)
    for f, t in zip(freq_indices_sorted, time_indices_sorted):
        ridge_mask[f, t] = True

    # Count ridges at each time point
    num_times = transform_shape[1]  # transform_shape = (scale, time)
    ridges_per_time = np.zeros(num_times, dtype=int)

    for t in range(num_times):
        ridges_per_time[t] = np.sum(ridge_mask[:, t])

    max_ridge_count = np.max(ridges_per_time)
    if max_ridge_count == 0:
        return np.array([]), []

    # Initialize index matrix and output arrays
    index_matrix = np.full((num_times, max_ridge_count), np.nan)

    # Create organized output arrays with same structure
    organized_arrays = []
    for arr in arrays_to_organize:
        if np.isrealobj(arr):
            organized_arrays.append(np.full((num_times, max_ridge_count), np.nan))
        else:
            organized_arrays.append(
                np.full((num_times, max_ridge_count), np.nan + 1j * np.nan)
            )

    # Fill index matrix with original indices
    ridge_counter = np.zeros(num_times, dtype=int)
    for i, (t, f) in enumerate(zip(time_indices_sorted, freq_indices_sorted)):
        ridge_idx = ridge_counter[t]
        index_matrix[t, ridge_idx] = i

        # Fill organized arrays with values
        for arr_idx, arr in enumerate(sorted_arrays):
            organized_arrays[arr_idx][t, ridge_idx] = arr[i]

        ridge_counter[t] += 1

    return index_matrix, organized_arrays


def create_3d_deviation_matrix(
    freq_matrix: np.ndarray,
    freq_next_pred_matrix: np.ndarray,
    freq_prev_pred_matrix: np.ndarray,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Create a 3D matrix of frequency deviations between ridge points at adjacent time steps.

    This function calculates the normalized frequency deviations between ridge points
    at adjacent time steps, using both forward and backward predictions.

    Parameters
    ----------
    freq_matrix : np.ndarray
        Matrix of ridge frequencies, shape (time, max_ridges)
    freq_next_pred_matrix : np.ndarray
        Matrix of predicted next frequencies, shape (time, max_ridges)
    freq_prev_pred_matrix : np.ndarray
        Matrix of predicted previous frequencies, shape (time, max_ridges)
    alpha : float, optional
        Maximum allowed frequency deviation (default: 0.25)

    Returns
    -------
    df : np.ndarray
        3D matrix with frequency deviations, shape (time-1, max_ridges, max_ridges)
        where df[t, i, j] is the deviation between ridge i at time t and ridge j at time t+1.
    """
    num_times, max_ridges = freq_matrix.shape

    # Create 3D matrices
    freq_3d = np.repeat(freq_matrix[:, :, np.newaxis], max_ridges, axis=2)
    freq_next_pred_3d = np.repeat(
        freq_next_pred_matrix[:, :, np.newaxis], max_ridges, axis=2
    )
    freq_prev_pred_3d = np.repeat(
        freq_prev_pred_matrix[:, :, np.newaxis], max_ridges, axis=2
    )

    # Shift and permute for next time comparison
    freq_shifted = np.roll(freq_3d, -1, axis=0)

    # Permute dimensions: (time, ridge, ridge) -> (time, ridge_next, ridge_current)
    freq_next_actual = np.transpose(freq_shifted, (0, 2, 1))

    # Forward prediction error
    with np.errstate(divide="ignore", invalid="ignore"):
        df1 = (freq_next_actual - freq_next_pred_3d) / freq_3d

    # Backward prediction error
    freq_prev_shifted = np.roll(freq_prev_pred_3d, -1, axis=0)
    freq_prev_next_actual = np.transpose(freq_prev_shifted, (0, 2, 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        df2 = (freq_prev_next_actual - freq_3d) / freq_3d

    # Combine bidirectional errors
    df = (np.abs(df1) + np.abs(df2)) / 2

    # Apply threshold mask
    df[df > alpha] = np.nan

    # Remove last time step (no next time to connect to)
    df = df[:-1, :, :]

    return df


def bilateral_minimum_selection(df: np.ndarray) -> np.ndarray:
    """
    This function uses a bilateral minimum selection approach to filter the 3D deviation matrix.
    It filters the matrix by selecting the minimum values in a way that respects both
    the row and column structure.

    Parameters:
    df : np.ndarray
        A 3D numpy array of shape (num_times, max_ridges, max_ridges) containing the deviation values.
    Returns:
    np.ndarray
        A filtered 3D numpy array where each slice contains the minimum values selected
        according to the bilateral minimum selection criteria.
    """
    num_times, max_ridges, _ = df.shape

    # Find which slices are valid (not all NaN)
    valid_rows = ~np.all(np.isnan(df), axis=2)

    df_step1 = np.full_like(df, np.nan)

    if np.any(valid_rows):
        # For each valid row, find minimum
        for t in range(num_times):
            for r in range(max_ridges):
                if valid_rows[t, r]:
                    min_idx = np.nanargmin(df[t, r, :])
                    df_step1[t, r, min_idx] = df[t, r, min_idx]

    # Similar approach for columns
    valid_cols = ~np.all(np.isnan(df_step1), axis=1)

    df_final = np.full_like(df_step1, np.nan)

    if np.any(valid_cols):
        for t in range(num_times):
            for s in range(max_ridges):
                if valid_cols[t, s]:
                    min_idx = np.nanargmin(df_step1[t, :, s])
                    df_final[t, min_idx, s] = df_step1[t, min_idx, s]

    return df_final


def separate_ridge_groups(
    freq_indices: NDArray[np.int_],
    time_indices: NDArray[np.int_],
    transform_shape: Tuple[int, int],
    ridge_quantity: NDArray[np.float64],
    freq_mesh: NDArray[np.float64],
    inst_frequency: NDArray[np.float64],
    inst_frequency_derivative: NDArray[np.float64],
    alpha: float = 1000.0,
    min_ridge_size: int = 3,
    max_gap: int = 2,
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """
    Separate ridge points into distinct groups using bidirectional frequency prediction
    and bilateral minimum selection for optimal assignment matching.

    Parameters
    ----------
    freq_indices : NDArray[np.int_]
        Array of frequency indices for ridge points
    time_indices : NDArray[np.int_]
        Array of time indices for ridge points
    transform_shape : Tuple[int, int]
        Shape of the original transform (scale, time)
    ridge_quantity : NDArray[np.float64]
        Ridge quantity used for matching points
    freq_mesh : NDArray[np.float64]
        Meshgrid of frequencies corresponding to the transform shape
    inst_frequency : NDArray[np.float64]
        Instantaneous frequency for each ridge point
    inst_frequency_derivative : NDArray[np.float64]
        Instantaneous frequency derivative for each ridge point
    alpha : float, optional
        Maximum allowed relative frequency difference for matching points (default: 1000.0)
    min_ridge_size : int, optional
        Minimum number of points for a valid ridge (default: 3)
    max_gap : int, optional
        Maximum allowed time gap between ridge points (default: 2)

    Returns
    -------
    ridge_data : Dict[int, Dict[str, Any]]
        Dictionary of ridges with indices and values for each ridge
        - 'indices': tuple of (freq_indices, time_indices)
        - 'values': list of arrays, each containing values at ridge points
    num_ridges : int
        Number of distinct ridges found
    """
    if len(freq_indices) == 0:
        return {}, 0

    ridge_interpolation_values = [
        freq_mesh,
        inst_frequency_derivative,
        ridge_quantity,
        inst_frequency,
    ]

    # Organize ridge points into time-ridge matrices
    index_matrix, organized_values = organize_ridge_points(
        freq_indices, time_indices, transform_shape, ridge_interpolation_values
    )

    # Extract organized arrays
    organized_freq = organized_values[0]
    organized_inst_freq_deriv = organized_values[1]

    # Get the sort index that was used inside organize_ridge_points
    # (necessary to associate "ridge point" with frequency)
    sort_idx = np.lexsort((freq_indices, time_indices))
    freq_indices_sorted = freq_indices[sort_idx]
    time_indices_sorted = time_indices[sort_idx]
    sorted_values = [val[sort_idx] for val in ridge_interpolation_values]

    # Predict next/prev frequencies using instantaneous frequency derivative
    organized_next_freq = organized_freq + organized_inst_freq_deriv
    organized_prev_freq = organized_freq - organized_inst_freq_deriv

    # Use the 3D deviation matrix approach instead of the previous method
    deviation_matrix_3d = create_3d_deviation_matrix(
        organized_freq, organized_next_freq, organized_prev_freq, alpha=alpha
    )

    # Apply bilateral minimum selection to get optimal connections
    filtered_deviation_matrix = bilateral_minimum_selection(deviation_matrix_3d)

    # Extract valid connections from the filtered matrix
    num_times, max_ridges = organized_freq.shape
    ridge_ids = np.full((num_times, max_ridges), -1, dtype=int)
    id_counter = 0

    # Initialize ridge IDs for first time step
    for r in range(max_ridges):
        if not np.isnan(organized_freq[0, r]):
            ridge_ids[0, r] = id_counter
            id_counter += 1

    # Propagate IDs based on bilateral minimum selection results
    for t in range(num_times - 1):
        connections = filtered_deviation_matrix[t]

        # Find valid connections (non-NaN values)
        valid_connections = np.where(~np.isnan(connections))

        if len(valid_connections[0]) > 0:
            source_ridges, target_ridges = valid_connections

            for src, tgt in zip(source_ridges, target_ridges):
                # Propagate ID from source ridge to target ridge
                if ridge_ids[t, src] >= 0:  # Source ridge has valid ID
                    if ridge_ids[t + 1, tgt] == -1:  # Target ridge not yet assigned
                        ridge_ids[t + 1, tgt] = ridge_ids[t, src]

        # Assign new IDs to unconnected ridges at next time step
        for r in range(max_ridges):
            if not np.isnan(organized_freq[t + 1, r]) and ridge_ids[t + 1, r] == -1:
                ridge_ids[t + 1, r] = id_counter
                id_counter += 1

    # Create groups from ridge IDs
    unique_ids_np = np.unique(ridge_ids)
    unique_ids_np = unique_ids_np[unique_ids_np >= 0]
    unique_ids = [int(uid) for uid in unique_ids_np]

    ridge_data = {}
    valid_ridge_count = 0

    for ridge_id in unique_ids:
        positions = np.argwhere(ridge_ids == ridge_id)

        if len(positions) < min_ridge_size:
            continue

        t_indices, r_indices = positions[:, 0], positions[:, 1]

        # Check for large time gaps
        if len(t_indices) > 1 and np.max(np.diff(np.sort(t_indices))) > max_gap:
            continue

        # Map back to original indices using index_matrix
        original_indices = []
        t_indices_int = [int(t) for t in t_indices]
        r_indices_int = [int(r) for r in r_indices]
        for t, r in zip(t_indices_int, r_indices_int):
            if np.isnan(index_matrix[t, r]):
                continue  # Skip if index is NaN
            idx = int(index_matrix[t, r])
            original_indices.append(idx)

        # Skip if too few original indices
        if len(original_indices) < min_ridge_size:
            continue

        # Get frequency and time indices from sorted arrays
        ridge_freq_indices = freq_indices_sorted[original_indices]
        ridge_time_indices = time_indices_sorted[original_indices]

        ridge_values = []
        for val_array in sorted_values:
            ridge_values.append(val_array[original_indices])

        valid_ridge_count += 1
        ridge_data[valid_ridge_count] = {
            "indices": (ridge_freq_indices, ridge_time_indices),
            "values": ridge_values,
        }

    return ridge_data, valid_ridge_count


def calculate_ridge_lengths(
    ridge_data: Dict[int, Dict[str, Any]], t: np.ndarray, num_ridges: int
) -> List[float]:
    """
    Calculate the length of each ridge group using Simpson's rule integration.

    Parameters
    ----------
    ridge_data : Dict[int, Dict[str, Any]]
        Dictionary of ridge data with indices and values for each ridge
    t : np.ndarray
        Time values corresponding to columns in the transform
    num_ridges : int
        Total number of ridges to process

    Returns
    -------
    List[float]
        List of calculated ridge lengths in periods
    """
    ridge_lengths = []

    # Process each ridge to calculate its length
    for ridge_id in range(1, num_ridges + 1):
        # Get ridge data
        if ridge_id not in ridge_data:
            ridge_lengths.append(0.0)
            continue

        # Extract frequency values and time indices
        freq_indices, time_indices = ridge_data[ridge_id]["indices"]
        freq_values = ridge_data[ridge_id]["values"][0]

        # Skip if ridge is empty
        if len(freq_indices) == 0:
            ridge_lengths.append(0.0)
            continue

        # Calculate the length of the ridge using Simpson's rule
        try:
            ridge_length = np.abs(integrate.simpson(freq_values, x=t[time_indices])) / (
                2.0 * np.pi
            )
            ridge_lengths.append(ridge_length)
        except Exception as e:
            print(f"Error calculating length for ridge {ridge_id}: {e}")
            ridge_lengths.append(0.0)

    return ridge_lengths


def ridge_analysis(
    wavelet: np.ndarray,
    freqs: np.ndarray,
    t: np.ndarray,
    ridge_type: str,
    amplitude_threshold: float = 0.1,
    min_ridge_size: int = 5,
    max_gap: int = 2,
) -> xr.Dataset:
    """
    Detect ridge points in wavelet transform, separate them into groups using frequency prediction,
    and calculate properties for each group using the coordinate-based format.

    Parameters
    ----------
    wavelet : np.ndarray
        Wavelet transform (complex-valued) (shape: scale, time)
    freqs : np.ndarray
        Frequency values corresponding to rows in the transform
    t : np.ndarray
        Time values corresponding to columns in the transform
    ridge_type : str
        Type of ridge detection ('amplitude' or 'phase')
    amplitude_threshold : float, optional
        Threshold for ridge detection, default=0.1
    min_ridge_size : int, optional
        Minimum number of points for a valid ridge, default=5
    max_gap : int, optional
        Maximum allowed time gap between ridge points, default=2

    Returns
    -------
    xr.Dataset
        Dataset containing ridge data as ragged arrays with all interpolated values
        and calculated properties:
        - 'time': time values for each ridge point
        - 'frequency': frequency values for each ridge point
        - 'inst_frequency': instantaneous frequency for each ridge point
        - 'inst_frequency_derivative': instantaneous frequency derivative for each ridge point
        - 'ridge_quantity': ridge quantity values for each ridge point
        - 'rowsize': number of points in each ridge group
        - 'ridge_length': length of each ridge group in periods
    """

    # Find ridge points
    ridge_points, ridge_quantity, processed_transform, inst_frequency = isridgepoint(
        wavelet_transform=wavelet,
        scale_frequencies=freqs,
        amplitude_threshold=amplitude_threshold,
        ridge_type=ridge_type,
    )

    # Create frequency meshgrid for interpolation
    freq_mesh = np.zeros_like(ridge_quantity)
    for i, f in enumerate(freqs):
        freq_mesh[i, :] = f

    # Calculate the derivative of the instantaneous frequency
    inst_frequency_derivative = np.gradient(inst_frequency, axis=-1)

    # Use ridge_shift_interpolation to get better accuracy and include derivative
    interpolation_arrays = ridge_shift_interpolation(
        ridge_points=ridge_points,
        ridge_quantity=ridge_quantity,
        y_arrays=[ridge_quantity, freq_mesh, inst_frequency, inst_frequency_derivative],
    )

    # Extract data from interpolation results
    freq_indices, time_indices = np.where(ridge_points)

    # Group the ridge points using frequency-based approach
    ridge_data, num_ridges = separate_ridge_groups(
        freq_indices=freq_indices,
        time_indices=time_indices,
        transform_shape=processed_transform.shape,
        ridge_quantity=interpolation_arrays[0],
        freq_mesh=interpolation_arrays[1],
        inst_frequency=interpolation_arrays[2],
        inst_frequency_derivative=interpolation_arrays[3],
        min_ridge_size=min_ridge_size,
        max_gap=max_gap,
    )

    # Calculate lengths for each ridge
    ridge_lengths = calculate_ridge_lengths(ridge_data, t, num_ridges)

    if num_ridges == 0:
        # Return empty dataset
        return xr.Dataset(
            {
                "time": (["obs"], np.array([])),
                "frequency": (["obs"], np.array([])),
                "inst_frequency": (["obs"], np.array([])),
                "inst_frequency_derivative": (["obs"], np.array([])),
                "ridge_quantity": (["obs"], np.array([])),
                "rowsize": (["id"], np.array([])),
                "ridge_length": (["id"], np.array([])),
            },
            coords={"obs": np.array([]), "id": np.array([])},
        )

    # Convert ridge_data to ragged array format
    # Sort ridge IDs to ensure consistent ordering
    ridge_ids = sorted(ridge_data.keys())

    all_times = []
    all_frequencies = []
    all_inst_frequencies = []
    all_inst_freq_derivatives = []
    all_ridge_quantities = []
    rowsizes = []

    # Process ridges in sorted order
    for ridge_id in ridge_ids:
        if ridge_id in ridge_data:
            # Get the indices for this ridge
            freq_indices_ridge, time_indices_ridge = ridge_data[ridge_id]["indices"]
            ridge_values = ridge_data[ridge_id]["values"]

            ridge_quantity_vals = ridge_values[2]
            frequency_vals = ridge_values[0]
            inst_frequency_vals = ridge_values[3]
            inst_freq_derivative_vals = ridge_values[1]

            # Get corresponding time values
            time_vals = t[time_indices_ridge]

            # Append to our lists (this creates an implicit ordering)
            all_times.extend(time_vals)
            all_frequencies.extend(frequency_vals)
            all_inst_frequencies.extend(inst_frequency_vals)
            all_inst_freq_derivatives.extend(inst_freq_derivative_vals)
            all_ridge_quantities.extend(ridge_quantity_vals)

            # Track ridge size
            rowsizes.append(len(time_indices_ridge))
        else:
            rowsizes.append(0)

    # Convert to numpy arrays
    all_times_np = np.array(all_times)
    all_frequencies_np = np.array(all_frequencies)
    all_inst_frequencies_np = np.array(all_inst_frequencies)
    all_inst_freq_derivatives_np = np.array(all_inst_freq_derivatives)
    all_ridge_quantities_np = np.array(all_ridge_quantities)
    rowsizes_np = np.array(rowsizes)

    # Create unique ridge IDs for coordinates
    unique_ridge_ids = np.array(ridge_ids)

    # Create the dataset
    ds = xr.Dataset(
        {
            "time": (["obs"], all_times_np),
            "frequency": (["obs"], all_frequencies_np),
            "inst_frequency": (["obs"], all_inst_frequencies_np),
            "inst_frequency_derivative": (["obs"], all_inst_freq_derivatives_np),
            "ridge_quantity": (["obs"], all_ridge_quantities_np),
            "rowsize": (["id"], rowsizes_np),
            "ridge_length": (["id"], ridge_lengths),
        },
        coords={"obs": np.arange(len(all_times_np)), "id": unique_ridge_ids},
    )

    # Add attributes
    ds["time"].attrs = {
        "long_name": "time",
        "data_type": "float64",
        "units": "Those corresponding to the wavelet transform",
        "description": "Time values corresponding to the wavelet transform",
    }
    ds["frequency"].attrs = {
        "long_name": "ridge frequency (interpolated)",
        "data_type": "float64",
        "units": "rad/[unit time]. Temporal units are those corresponding to the wavelet transform",
        "description": "Frequency values corresponding to the wavelet transform",
    }
    ds["inst_frequency"].attrs = {
        "long_name": "instantaneous frequency (interpolated)",
        "data_type": "float64",
        "units": "rad/[unit time]. Temporal units are those corresponding to the wavelet transform",
        "description": "Instantaneous frequency calculated from wavelet phase",
    }
    ds["inst_frequency_derivative"].attrs = {
        "long_name": "instantaneous frequency derivative (interpolated)",
        "data_type": "float64",
        "units": "rad/[unit time]^2. Temporal units are those corresponding to the wavelet transform",
        "description": "Time derivative of instantaneous frequency",
    }
    ds["ridge_quantity"].attrs = {
        "long_name": "Chosen ridge quantity value (interpolated)",
        "data_type": "float64",
        "description": f"Quantity used for {ridge_type} ridge detection",
    }
    ds["rowsize"].attrs = {
        "long_name": "number of observations per ridge",
        "data_type": "int",
        "description": "number of time points in each ridge",
    }
    ds["ridge_length"].attrs = {
        "long_name": "ridge length",
        "data_type": "float64",
        "units": "periods",
        "description": "ridge length calculated using Simpson rule integration",
    }

    # Global attributes
    ds.attrs = {
        "title": "Wavelet Ridge Analysis Results",
        "ridge_type": ridge_type,
        "amplitude_threshold": amplitude_threshold,
        "min_ridge_size": min_ridge_size,
        "max_gap": max_gap,
        "num_ridges": num_ridges,
        "description": "Ridge analysis results in ragged array format with interpolated values",
    }

    return ds

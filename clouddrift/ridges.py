from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
import warnings

def gradient_of_angle(
    x: NDArray[np.float64 | np.complex128 | np.float32 | np.complex64],
    edge_order: int = 1,
    axis: int = -1,
    discont: float = np.pi
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

    result[tuple(slice_interior_dest_of_source_even)] = np.diff(np.unwrap(
        x[tuple(slice_interior_source_even)], axis=axis, discont=discont), axis=axis) / 2
    result[tuple(slice_interior_dest_of_source_odd)] = np.diff(np.unwrap(
        x[tuple(slice_interior_source_odd)], axis=axis, discont=discont), axis=axis) / 2

    slice_begin_dest[axis] = slice(0, 1)
    slice_end_dest[axis] = slice(-1, None)
    if edge_order == 1:
        slice_begin_source[axis] = slice(0, 2)
        slice_end_source[axis] = slice(-2, None)
        result[tuple(slice_begin_dest)] = np.diff(np.unwrap(
            x[tuple(slice_begin_source)], axis=axis, discont=discont), axis=axis)
        result[tuple(slice_end_dest)] = np.diff(np.unwrap(
            x[tuple(slice_end_source)], axis=axis, discont=discont), axis=axis)
    elif edge_order == 2:
        slice_begin_source[axis] = slice(1, 2)
        slice_end_source[axis] = slice(-2, -1)
        begin_source = np.concatenate([x[tuple(slice_end_dest)], x[tuple(slice_begin_source)]], axis=axis)
        end_source = np.concatenate([x[tuple(slice_end_source)], x[tuple(slice_begin_dest)]], axis=axis)
        result[tuple(slice_begin_dest)] = np.diff(np.unwrap(
            begin_source, axis=axis, discont=discont), axis=axis) / 2
        result[tuple(slice_end_dest)] = np.diff(np.unwrap(
            end_source, axis=axis, discont=discont), axis=axis) / 2
    else:
        raise ValueError('Unexpected edge_order: {}'.format(edge_order))

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
        Input signal array (complex-valued)
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
    amplitude = np.abs(signal)

    # Calculate instantaneous frequency
    omega = gradient_of_angle(np.angle(signal), axis=axis) * sample_rate

    # Calculate instantaneous bandwidth
    log_amplitude = np.log(amplitude)
    upsilon = np.gradient(log_amplitude, axis=axis) * sample_rate

    # Calculate instantaneous curvature
    d_omega = np.gradient(omega, axis=axis) * sample_rate
    d_upsilon = np.gradient(upsilon, axis=axis) * sample_rate
    xi = upsilon**2 + d_upsilon + 1j * d_omega

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
        of signals across which to compute joint moments.
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

    # Calculate instantaneous moments
    if wavelet_transform.ndim > 2 and wavelet_transform.shape[2] > 1:
        # Multivariate case
        amplitude, inst_frequency = instmom_multivariate(
            wavelet_transform, time_axis=0, joint_axis=2
        )[:2]
    else:
        # Univariate case
        amplitude, inst_frequency = instmom_univariate(wavelet_transform, axis=0)[:2]

    # Determine ridge quantity based on ridge type
    if ridge_type.lower().startswith("amp"):
        ridge_quantity = amplitude
    else:  # phase-based ridges
        # Create array of scale frequencies matching time dimension
        freq_matrix = np.broadcast_to(
            scale_frequencies[np.newaxis, :].T,
            (wavelet_transform.shape[0], wavelet_transform.shape[1]),
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
        # For phase ridges: find zero crossings with negative slope in time
        for i in range(1, ridge_quantity.shape[0] - 1):
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


def separate_ridge_groups_dbscan(
    ridge_points: NDArray[np.bool_],
    eps: float = 5.0,
    min_samples: int = 3,
    scale_factor: float = 1.0,
) -> Tuple[NDArray[np.int_], int]:
    """
    Separate ridge points into distinct groups using DBSCAN clustering.

    This function identifies separate clusters of ridge points in the time-frequency
    plane using a distance-based clustering algorithm, which can handle points that
    are not directly adjacent.

    Parameters
    ----------
    ridge_points : NDArray[np.bool_]
        Boolean array where True indicates ridge points, with shape (frequency, time)
    eps : float, optional
        Maximum distance between points to be considered in the same cluster (default: 5.0)
    min_samples : int, optional
        Minimum number of points required to form a cluster (default: 3)
    scale_factor : float, optional
        Factor to scale frequency dimension relative to time dimension (default: 1.0)
        Higher values make frequency differences more important than time differences

    Returns
    -------
    labeled_ridges : NDArray[np.int_]
        Integer array where values indicate group membership (0 is background)
    num_groups : int
        Number of distinct ridge groups found

    Notes
    -----
    This function uses scikit-learn's DBSCAN algorithm which can connect points that
    are not directly adjacent. This is more flexible than connected component analysis
    but may be less efficient for large datasets.
    """
    # Get indices of ridge points
    freq_indices, time_indices = np.where(ridge_points)

    # Skip if no ridge points
    if len(freq_indices) == 0:
        return np.zeros_like(ridge_points, dtype=int), 0

    # Scale frequency dimension if needed
    if scale_factor != 1.0:
        freq_scaled = freq_indices * scale_factor
    else:
        freq_scaled = freq_indices.astype(float)

    # Combine indices into points
    points = np.column_stack([time_indices, freq_scaled])

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # Create labeled ridge array
    labeled_ridges = np.zeros_like(ridge_points, dtype=int)
    for i, (f, t) in enumerate(zip(freq_indices, time_indices)):
        if labels[i] >= 0:  # Ignore noise points (-1)
            labeled_ridges[f, t] = labels[i] + 1  # Add 1 so background is 0

    # Count number of clusters (excluding noise points)
    num_clusters = len(set(labels) - {-1})

    return labeled_ridges, num_clusters


def ridge_shift_interpolation(
    ridge_points: NDArray[np.bool_],
    ridge_quantity: NDArray[np.float64],
    y_arrays: List[NDArray[np.float64]],
) -> Dict[str, Any]:
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
    ridge_data : dict
        Dictionary containing:
        - 'indices': tuple of (freq_indices, time_indices)
        - 'values': list of arrays, each containing interpolated values at ridge points
        - 'shape': original array shape for reconstruction
    """
    # Skip if no ridge points
    if not np.any(ridge_points):
        # Return empty result with proper structure
        return {
            'indices': (np.array([], dtype=int), np.array([], dtype=int)),
            'values': [np.array([], dtype=arr.dtype) for arr in y_arrays],
            'shape': ridge_points.shape
        }
        
    # Get indices of ridge points
    freq_indices, time_indices = np.where(ridge_points)
    
    # Initialize output value arrays (1D arrays for each y_array's ridge points)
    y_values_interpolated = [np.zeros(len(freq_indices), dtype=y_array.dtype) for y_array in y_arrays]
    
    # Create masks for boundary points
    boundary_mask = (freq_indices <= 0) | (freq_indices >= ridge_quantity.shape[0] - 1)
    interior_mask = ~boundary_mask
    
    # Process boundary points (just copy original values)
    if np.any(boundary_mask):
        boundary_freq = freq_indices[boundary_mask]
        boundary_time = time_indices[boundary_mask]
        for j, y_array in enumerate(y_arrays):
            y_values_interpolated[j][boundary_mask] = y_array[boundary_freq, boundary_time]
    
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
                    w = (lin_prev[prev_higher][nonzero] - lin_curr[prev_higher][nonzero]) / denom[nonzero]
                    weights[prev_higher][nonzero] = np.minimum(0.5, w)
            
            # For points where next is higher
            next_higher = ~prev_higher
            if np.any(next_higher):
                # Avoid division by zero
                denom = lin_next[next_higher] - lin_curr[next_higher]
                nonzero = denom != 0
                if np.any(nonzero):
                    w = (lin_next[next_higher][nonzero] - lin_curr[next_higher][nonzero]) / denom[nonzero]
                    weights[next_higher][nonzero] = np.minimum(0.5, w)
            
            # Apply linear interpolation for each y_array
            for j, y_array in enumerate(y_arrays):
                y_interp = np.zeros_like(lin_curr, dtype=y_array.dtype)
                
                # Points where previous is higher
                if np.any(prev_higher):
                    prev_weights = weights[prev_higher]
                    y_curr = y_array[lin_freq[prev_higher], lin_time[prev_higher]]
                    y_prev = y_array[lin_freq[prev_higher] - 1, lin_time[prev_higher]]
                    y_interp[prev_higher] = (1 - prev_weights) * y_curr + prev_weights * y_prev
                
                # Points where next is higher
                if np.any(next_higher):
                    next_weights = weights[next_higher]
                    y_curr = y_array[lin_freq[next_higher], lin_time[next_higher]]
                    y_next = y_array[lin_freq[next_higher] + 1, lin_time[next_higher]]
                    y_interp[next_higher] = (1 - next_weights) * y_curr + next_weights * y_next
                
                # Store the interpolated values in the result array
                y_values_interpolated[j][lin_indices] = y_interp
    
    # Return results in coordinate format
    return {
        'indices': (freq_indices, time_indices),
        'values': y_values_interpolated,
        'shape': ridge_points.shape
    }

def organize_ridge_points(
    freq_indices: NDArray[np.int_],
    time_indices: NDArray[np.int_],
    transform_shape: Tuple[int, int],
    arrays_to_organize: List[NDArray[np.float64]]
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
        Shape of the wavelet transform (frequency, time)
    arrays_to_organize : list
        List of arrays containing data for each ridge point
        
    Returns
    -------
    index_matrix : NDArray[np.float64]
        Matrix of shape (time, max_ridges) mapping original indices to time-ridge coordinates
    organized_arrays : list
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
    num_times = transform_shape[1]  # transform_shape = (freq, time)
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
            organized_arrays.append(np.full((num_times, max_ridge_count), 
                                          np.nan + 1j*np.nan))
    
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


def calculate_frequency_deviations(
    organized_freq: NDArray[np.float64],
    organized_next_freq_predictions: NDArray[np.float64], 
    organized_prev_freq_predictions: NDArray[np.float64],
    alpha: float = 0.25
) -> NDArray[np.float64]:
    """
    Calculate normalized frequency deviations between ridge points at adjacent time steps.
    
    Parameters
    ----------
    organized_freq : NDArray[np.float64]
        Matrix of ridge frequencies, shape (time, max_ridges)
    organized_next_freq_predictions : NDArray[np.float64]
        Matrix of predicted next frequencies, shape (time, max_ridges)
    organized_prev_freq_predictions : NDArray[np.float64]
        Matrix of predicted previous frequencies, shape (time, max_ridges)
    alpha : float
        Maximum allowed frequency deviation (default: 0.25)
        
    Returns
    -------
    deviation_matrix : NDArray[np.float64]
        Matrix with lowest deviation for each ridge connection
        Shape (time-1, max_ridges, max_ridges)
    """
    num_times, max_ridges = organized_freq.shape

    # Initialize a sparse frequency deviation matrix
    deviation_matrix = np.full((num_times-1, max_ridges, max_ridges), np.nan)
    
    for t in range(0, num_times-1):
        # Get frequencies at current, next, and previous time
        current_freq = np.real(organized_freq[t, :])
        next_freq = np.real(organized_freq[t+1, :]) if t < num_times - 1 else np.array([np.nan, np.nan], dtype=float)
        prev_freq = np.real(organized_freq[t-1, :]) if t > 0 else np.array([np.nan, np.nan], dtype=float)

        # Get predicted frequencies
        next_freq_pred = organized_next_freq_predictions[t, :]
        prev_freq_pred = organized_prev_freq_predictions[t, :]

        current_indices = np.where(current_freq)[0]
        next_indices = np.where(next_freq)[0]

        combined_deviation = np.full((max_ridges, max_ridges), np.nan)

        if np.any(current_freq) and np.any(next_freq):
            # Forward prediction
            forward_diff = np.abs(next_freq_pred[:, np.newaxis] - next_freq[np.newaxis, :])
            norm_forward = forward_diff / current_freq[:, np.newaxis]

            # Backward prediction
            if t > 0 and prev_freq.size > 0 and prev_freq_pred.size > 0:
                backward_diff = np.abs(prev_freq_pred[np.newaxis, :] - prev_freq[:, np.newaxis])
                norm_backward = backward_diff / current_freq[:, np.newaxis]
            else:
                norm_backward = np.zeros_like(norm_forward)

            # Average the bidirectional deviations
            local_combined = (norm_forward + norm_backward) / 2.0
            local_combined[local_combined > alpha] = np.nan

            for i, curr_idx in enumerate(current_indices):
                for j, next_idx in enumerate(next_indices):
                    combined_deviation[curr_idx, next_idx] = local_combined[i, j]

            
        deviation_matrix[t] = combined_deviation

    return deviation_matrix
        

def separate_ridge_groups_frequency(
    ridge_data: Dict[str, Any],
    alpha: float = 1000.0,
    min_group_size: int = 3,
    max_gap: int = 2
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """
    Separate ridge points into distinct groups using bidirectional frequency prediction
    and optimal assignment matching.

    Parameters
    ----------
    ridge_data : Dict[str, Any]
        Dictionary from ridge_shift_interpolation containing ridge point data
    alpha : float, optional
        Maximum allowed relative frequency difference for matching points (default: 0.1)
    min_group_size : int, optional
        Minimum number of points for a valid group (default: 3)
    max_gap : int, optional
        Maximum allowed time gap between ridge points (default: 2)

    Returns
    -------
    group_data : Dict[int, Dict[str, Any]]
        Dictionary of groups with indices and values for each group
    num_groups : int
        Number of distinct ridge groups found
    """
    freq_indices, time_indices = ridge_data['indices']
    values = ridge_data['values']

    if len(freq_indices) == 0:
        return {}, 0

    # Organize ridge points into time-ridge matrices
    index_matrix, [organized_power, organized_freq, organized_inst_freq, organized_inst_freq_deriv] = organize_ridge_points(
        freq_indices, time_indices, ridge_data['shape'], ridge_data['values']
    )

    # Get the sort index that was used inside organize_ridge_points 
    # (necessary to associate "ridge point" with frequency)
    sort_idx = np.lexsort((freq_indices, time_indices))
    freq_indices_sorted = freq_indices[sort_idx]
    time_indices_sorted = time_indices[sort_idx]
    sorted_values = [val[sort_idx] for val in values]

    # Predict next/prev frequencies using instantaneous frequency derivative
    organized_next_freq = organized_freq + organized_inst_freq_deriv
    organized_prev_freq = organized_freq - organized_inst_freq_deriv

    # Calculate frequency deviations
    deviation_matrix = calculate_frequency_deviations(
        organized_freq, organized_next_freq, organized_prev_freq, alpha=alpha
    )

    # Assignment and ID propagation
    num_times, max_ridges = organized_freq.shape
    ridge_ids = np.full((num_times, max_ridges), -1, dtype=int)
    id_counter = 0
    for t in range(num_times):
        for r in range(max_ridges):
            if not np.isnan(organized_freq[t, r]):
                ridge_ids[t, r] = id_counter
                id_counter += 1

    large_cost = np.nanmax(deviation_matrix) + 100.0

    for t in range(num_times - 1):
        cost = deviation_matrix[t]
        if np.all(np.isnan(cost)):
            continue
        cost = np.where(np.isnan(cost), large_cost, cost)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < large_cost:
                ridge_ids[t + 1, j] = ridge_ids[t, i]


    # Create groups from ridge IDs
    unique_ids = np.unique(ridge_ids)
    unique_ids = unique_ids[unique_ids >= 0]  # Filter out -1 (no ridge)
    
    group_data = {}
    valid_group_count = 0
    
    for group_id in unique_ids:
        positions = np.argwhere(ridge_ids == group_id)
        
        if len(positions) < min_group_size:
            continue
            
        t_indices, r_indices = positions[:, 0], positions[:, 1]
        
        # Map back to original indices using index_matrix
        original_indices = []
        for t, r in zip(t_indices, r_indices):
            if np.isnan(index_matrix[t, r]):
                continue  # Skip if index is NaN
            idx = int(index_matrix[t, r])
            original_indices.append(idx)
        
        # Skip if too few original indices
        if len(original_indices) < min_group_size:
            continue

        # Skip if gap is too large
        if np.max(np.diff(t_indices)) > max_gap:
            continue
            
        # Get frequency and time indices from sorted arrays
        group_freq_indices = freq_indices_sorted[original_indices]
        group_time_indices = time_indices_sorted[original_indices]
        
        group_values = []
        for val_array in sorted_values:
            group_values.append(val_array[original_indices])

        valid_group_count += 1
        group_data[valid_group_count] = {
            "indices": (group_freq_indices, group_time_indices),
            "values": group_values
        }
    
    return group_data, valid_group_count


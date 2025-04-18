from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment


def instmom_univariate(
    signal: NDArray[np.complex128], sample_rate: float = 1.0, axis: int = 0
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
    phase = np.angle(signal)
    unwrapped_phase = np.unwrap(phase, axis=axis)
    omega = np.gradient(unwrapped_phase, axis=axis) * sample_rate

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
    time_axis: int = 0,
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
    unwrapped_phase = np.unwrap(
        phase, axis=time_axis
    )  # Unwrap phase to prevent 2π jumps
    omega = np.gradient(unwrapped_phase, axis=time_axis) * sample_rate

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
) -> dict:
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


def get_group_data(
    group_data: List[dict], 
    group_id: int, 
    data_type: int = 0
) -> Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float64]]:
    """
    Extract data for a specific group in a usable format.
    
    Parameters
    ----------
    group_data : list
        List of group data dictionaries
    group_id : int
        Group ID (1-based)
    data_type : int, optional
        Data type to extract:
        0 = ridge quantity (power)
        1 = frequency
        2 = instantaneous frequency
        
    Returns
    -------
    freq_indices : ndarray
        Frequency indices
    time_indices : ndarray
        Time indices
    values : ndarray
        Values for the requested data type
    """
    if group_id < 1 or group_id > len(group_data):
        raise ValueError(f"Group ID {group_id} out of range (1-{len(group_data)})")
        
    group_dict = group_data[group_id-1]
    freq_indices, time_indices = group_dict['indices']
    values = group_dict['values'][data_type]
    
    return freq_indices, time_indices, values

def separate_ridge_groups_frequency(
    ridge_data: dict,
    alpha: float = 0.1,
    min_group_size: int = 3,
    max_gap: int = 2
) -> Tuple[dict, int]:
    """
    Separate ridge points into distinct groups using bidirectional frequency prediction
    and optimal assignment matching.
    
    Parameters
    ----------
    ridge_data : dict
        Dictionary from ridge_shift_interpolation containing ridge point data
    alpha : float, optional
        Maximum allowed relative frequency difference for matching points (default: 0.1)
    min_group_size : int, optional
        Minimum number of points for a valid group (default: 3)
    max_gap : int, optional
        Maximum allowed time gap between ridge points (default: 2)
    
    Returns
    -------
    group_data : dict
        Dictionary of groups with indices and values for each group
    num_groups : int
        Number of distinct ridge groups found
    """
    
    # Extract ridge point information
    freq_indices, time_indices = ridge_data['indices']
    values = ridge_data['values']
    
    # Debug: print basic information
    print(f"Total ridge points: {len(freq_indices)}")
    print(f"Time range: {min(time_indices)} to {max(time_indices)}")
    
    # If there are no ridge points, return empty result
    if len(freq_indices) == 0:
        return {}, 0
        
    # Get physical values at ridge points
    power = values[0]
    actual_freq = values[1]
    
    # Debug: frequency values
    print(f"Frequency range: {np.min(actual_freq):.2f} to {np.max(actual_freq):.2f}")
    
    # Create time-to-points mapping for efficient lookup
    time_to_points = {}
    for i, t in enumerate(time_indices):
        if t not in time_to_points:
            time_to_points[t] = []
        time_to_points[t].append(i)
    
    # Debug: time point distribution
    time_counts = [len(pts) for t, pts in time_to_points.items()]
    print(f"Points per time step - min: {min(time_counts)}, max: {max(time_counts)}, avg: {np.mean(time_counts):.1f}")
    
    # Initialize ridge group labels
    group_labels = np.full(len(freq_indices), -1, dtype=int)
    next_group_id = 1
    
    # Process time steps in order
    sorted_times = sorted(time_to_points.keys())
    for i, current_time in enumerate(sorted_times):
        # Find next valid time step within max_gap
        next_time = None
        for gap in range(1, max_gap + 1):
            if i + gap < len(sorted_times):
                next_time = sorted_times[i + gap]
                break
        
        if next_time is None:
            continue
            
        # Get points at these time steps
        curr_idxs = time_to_points[current_time]
        next_idxs = time_to_points[next_time]
        
        # Debug: check time steps
        if not curr_idxs or not next_idxs:
            print(f"Empty set of points at time {current_time} or {next_time}")
            continue
        
        # Create cost matrix
        cost_matrix = np.full((len(curr_idxs), len(next_idxs)), np.inf)
        
        # Fill cost matrix with relative frequency differences
        for ci, curr_idx in enumerate(curr_idxs):
            current_freq = actual_freq[curr_idx]
            
            if current_freq <= 0:
                continue
                
            for ni, next_idx in enumerate(next_idxs):
                next_freq = actual_freq[next_idx]
                
                # Calculate relative frequency difference
                rel_diff = abs(next_freq - current_freq) / current_freq
                
                # Store cost if within tolerance
                if rel_diff <= alpha:
                    cost_matrix[ci, ni] = rel_diff
        
        # Debug: check if any matches are possible
        valid_matches = np.isfinite(cost_matrix).any()
        if not valid_matches:
            print(f"No valid matches between time {current_time} and {next_time} (alpha={alpha})")
            continue
        
        try:
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Filter valid matches
            valid_indices = np.isfinite(cost_matrix[row_ind, col_ind])
            valid_row_ind = row_ind[valid_indices]
            valid_col_ind = col_ind[valid_indices]
            
            # Assign or propagate group IDs
            for ri, ci in zip(valid_row_ind, valid_col_ind):
                curr_idx = curr_idxs[ri]
                next_idx = next_idxs[ci]
                
                if group_labels[curr_idx] == -1:
                    # Start new group
                    group_labels[curr_idx] = next_group_id
                    group_labels[next_idx] = next_group_id
                    next_group_id += 1
                else:
                    # Propagate existing group
                    group_labels[next_idx] = group_labels[curr_idx]
        
        except ValueError as e:
            print(f"Error at time {current_time}: {e}")
            print(f"Cost matrix shape: {cost_matrix.shape}, min: {np.min(cost_matrix)}, finite values: {np.isfinite(cost_matrix).sum()}")
            continue
    
    # Filter groups by size and create output
    groups = {}
    for group_id in range(1, next_group_id):
        group_mask = (group_labels == group_id)
        if np.sum(group_mask) >= min_group_size:
            group_freq_indices = freq_indices[group_mask]
            group_time_indices = time_indices[group_mask]
            
            group_values = []
            for val_array in values:
                group_values.append(val_array[group_mask])
            
            groups[group_id] = {
                'indices': (group_freq_indices, group_time_indices),
                'values': group_values,
                'shape': ridge_data['shape']
            }
    
    print(f"Created {len(groups)} groups (min size: {min_group_size})")
    return groups, len(groups)
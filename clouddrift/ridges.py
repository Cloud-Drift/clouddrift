import numpy as np
from clouddrift.wavelet import (
    morse_wavelet_transform,
    morse_logspace_freq,
)
from typing import Optional, Union, Tuple, List
from sklearn.cluster import DBSCAN
from numpy.typing import NDArray


def instmom_univariate(
        signal: NDArray[np.complex128], 
        sample_rate: float = 1.0, 
        axis: int = 0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    '''
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
    '''
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
        joint_axis: int = -1
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''
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
    '''    
    # Calculate univariate moments for each signal
    amplitude = np.abs(signals)
    
    # Calculate instantaneous radian frequency for each signal
    phase = np.angle(signals)
    unwrapped_phase = np.unwrap(phase, axis=time_axis) # Unwrap phase to prevent 2π jumps
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
    joint_amplitude = np.sqrt(np.mean(squared_amplitude, axis=joint_axis, keepdims=True))
    
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
    joint_upsilon = np.sqrt(np.sum(np.abs(deviation)**2 * normalized_weights, 
                                  axis=joint_axis, keepdims=True))
    
    # Calculate joint curvature (fourth moment)
    curvature_deviation = xi + 2j * upsilon * (omega - omega_mean) - (omega - omega_mean)**2
    joint_xi = np.sqrt(np.sum(np.abs(curvature_deviation)**2 * normalized_weights,
                             axis=joint_axis, keepdims=True))
    
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
    mask: Optional[NDArray[np.bool_]] = None
) -> Tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.complex128], NDArray[np.float64]]:
    '''
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
    '''
    
    # Calculate instantaneous moments
    if wavelet_transform.ndim > 2 and wavelet_transform.shape[2] > 1:
        # Multivariate case
        amplitude, inst_frequency = instmom_multivariate(
            wavelet_transform, time_axis=0, joint_axis=2
        )[:2]
    else:
        # Univariate case
        amplitude, inst_frequency = instmom_univariate(
            wavelet_transform, axis=0
        )[:2]
    
    # Determine ridge quantity based on ridge type
    if ridge_type.lower().startswith('amp'):
        ridge_quantity = amplitude
    else:  # phase-based ridges
        # Create array of scale frequencies matching time dimension
        freq_matrix = np.broadcast_to(
            scale_frequencies[np.newaxis, :].T, 
            (wavelet_transform.shape[0], wavelet_transform.shape[1])
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
        phase_average[mask_nonzero] = weight_sum[mask_nonzero] / weight_total[mask_nonzero]
        
        # Create joint transform with combined magnitude and average phase
        joint_magnitude = np.sqrt(np.sum(transform_magnitude**2, axis=2))
        joint_phase = np.angle(phase_average)
        processed_transform = joint_magnitude * np.exp(1j * joint_phase)
    else:
        processed_transform = wavelet_transform
    
    # Define ridge points array
    ridge_points = np.zeros(ridge_quantity.shape, dtype=bool)
    
    # Process all time points except edges
    if ridge_type.lower().startswith('amp'):
        # For amplitude ridges: find local maxima in time
        for i in range(1, ridge_quantity.shape[0]-1):
            ridge_points[i, :] = (
                (ridge_quantity[i-1, :] < ridge_quantity[i, :]) & 
                (ridge_quantity[i+1, :] < ridge_quantity[i, :])
            )
    else:
        # For phase ridges: find zero crossings with negative slope in time
        for i in range(1, ridge_quantity.shape[0]-1):
            ridge_points[i, :] = (
                ((ridge_quantity[i-1, :] < 0) & (ridge_quantity[i+1, :] >= 0)) | 
                ((ridge_quantity[i-1, :] <= 0) & (ridge_quantity[i+1, :] > 0))
            )
    
    # Ensure we have the strongest local extrema
    error_matrix = np.abs(ridge_quantity)
    
    # Remove points where error is larger than adjacent time points
    for i in range(1, ridge_quantity.shape[0]-1):
        # Check previous time step
        if i > 1:
            is_prev_ridge = ridge_points[i-1, :]
            is_bigger_than_prev = error_matrix[i, :] > error_matrix[i-1, :]
            ridge_points[i, :] = ridge_points[i, :] & ~(is_prev_ridge & is_bigger_than_prev)
        
        # Check next time step
        if i < ridge_quantity.shape[0]-2:
            is_next_ridge = ridge_points[i+1, :]
            is_bigger_than_next = error_matrix[i, :] > error_matrix[i+1, :]
            ridge_points[i, :] = ridge_points[i, :] & ~(is_next_ridge & is_bigger_than_next)
    
    # Apply basic filtering criteria
    ridge_points = ridge_points & ~np.isnan(processed_transform)
    ridge_points = ridge_points & (np.abs(processed_transform) >= amplitude_threshold)
    
    # Apply frequency constraints if provided
    if freq_min is not None and freq_max is not None:
        # Create frequency constraint matrices
        if np.isscalar(freq_min) or (isinstance(freq_min, np.ndarray) and freq_min.size == 1):
            freq_min_matrix = np.full(processed_transform.shape[:2], freq_min)
        else:
            # Expand row vector to match dimensions
            freq_min_matrix = np.broadcast_to(
                freq_min[:, np.newaxis], 
                processed_transform.shape[:2]
            )
            
        if np.isscalar(freq_max) or (isinstance(freq_max, np.ndarray) and freq_max.size == 1):
            freq_max_matrix = np.full(processed_transform.shape[:2], freq_max)
        else:
            # Expand row vector to match dimensions
            freq_max_matrix = np.broadcast_to(
                freq_max[:, np.newaxis], 
                processed_transform.shape[:2]
            )
        
        # Apply frequency constraints
        freq_constraint = (inst_frequency > freq_min_matrix) & (inst_frequency < freq_max_matrix)
        ridge_points = ridge_points & freq_constraint
    
    # Apply additional mask if provided
    if mask is not None:
        ridge_points = ridge_points & mask
    
    return ridge_points, ridge_quantity, processed_transform, inst_frequency


def separate_ridge_groups_dbscan(
    ridge_points: NDArray[np.bool_], 
    eps: float = 5.0, 
    min_samples: int = 3,
    scale_factor: float = 1.0
) -> Tuple[NDArray[np.int_], int]:
    '''
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
    '''
    # Get indices of ridge points
    freq_indices, time_indices = np.where(ridge_points)
    
    # Skip if no ridge points
    if len(freq_indices) == 0:
        return np.zeros_like(ridge_points, dtype=int), 0
    
    # Scale frequency dimension if needed
    if scale_factor != 1.0:
        freq_scaled = freq_indices * scale_factor
    else:
        freq_scaled = freq_indices
    
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
) -> List[NDArray[np.float64]]:
    '''
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
    y_arrays_interpolated : List[NDArray[np.float64]]
        List of arrays interpolated at the optimal scale values.
    '''
    
    # Get indices of ridge points
    freq_indices, time_indices = np.where(ridge_points)

    # Skip if no ridge points
    if len(freq_indices) == 0:
        return y_arrays
    
    # Initialize list for interpolated y_arrays
    y_arrays_interpolated = [np.full_like(y_array, np.nan) for y_array in y_arrays]

    # Iterate over each ridge point
    for i, (f_idx, t_idx) in enumerate(zip(freq_indices, time_indices)):
        
        # Skip points at frequency boundaries
        if f_idx <= 0 or f_idx >= ridge_quantity.shape[0]-1:
            # Keep original values at boundaries
            for j, y_array in enumerate(y_arrays):
                y_arrays_interpolated[j][f_idx, t_idx] = y_array[f_idx, t_idx]
            continue
        
        # Get ridge quantity at current point and neighbors
        ridge_prev = ridge_quantity[f_idx-1, t_idx]
        ridge_curr = ridge_quantity[f_idx, t_idx]
        ridge_next = ridge_quantity[f_idx+1, t_idx]
        
        # Fit quadratic: y = ax² + bx + c where x = {-1, 0, 1} for the three points
        a = 0.5 * (ridge_next + ridge_prev - 2*ridge_curr)
        b = 0.5 * (ridge_next - ridge_prev)
        
        # If a < 0, there's a maximum to find
        if a < 0:
            # Find location of maximum: x = -b/(2a)
            x_max = -b/(2*a)
            
            # Only use if it's within our range [-1, 1]
            if -1 <= x_max <= 1:
                
                # Interpolate each y_array using the same quadratic approach
                for j, y_array in enumerate(y_arrays):
                    y_prev = y_array[f_idx-1, t_idx]
                    y_curr = y_array[f_idx, t_idx]
                    y_next = y_array[f_idx+1, t_idx]
                    
                    # Quadratic coefficients for this array
                    y_a = 0.5 * (y_next + y_prev - 2*y_curr)
                    y_b = 0.5 * (y_next - y_prev)
                    y_c = y_curr
                    
                    # Evaluate quadratic at the interpolated position
                    y_interp = y_a * x_max**2 + y_b * x_max + y_c
                    y_arrays_interpolated[j][f_idx, t_idx] = y_interp
                
                continue
        
        # If quadratic interpolation didn't work, try linear
        # Check which neighbor has higher value
        if ridge_prev > ridge_curr or ridge_next > ridge_curr:
            if ridge_prev >= ridge_next:
                # Previous point is higher
                if ridge_prev == ridge_curr:
                    weight = 0
                else:
                    # Linear interpolation toward the previous point
                    weight = (ridge_prev - ridge_curr) / (ridge_prev - ridge_curr + 1e-10)
                    weight = min(0.5, weight)  # Limit the weight for stability
                
                # Interpolate each y_array linearly
                for j, y_array in enumerate(y_arrays):
                    if weight == 0:
                        y_interp = y_array[f_idx, t_idx]
                    else:
                        y_interp = (1-weight) * y_array[f_idx, t_idx] + weight * y_array[f_idx-1, t_idx]
                    y_arrays_interpolated[j][f_idx, t_idx] = y_interp
                
            else:
                # Next point is higher
                if ridge_next == ridge_curr:
                    # Equal values, don't interpolate
                    weight = 0
                else:
                    # Linear interpolation toward the next point
                    weight = (ridge_next - ridge_curr) / (ridge_next - ridge_curr + 1e-10)
                    weight = min(0.5, weight)  # Limit the weight for stability
                
                # Interpolate each y_array linearly
                for j, y_array in enumerate(y_arrays):
                    if weight == 0:
                        y_interp = y_array[f_idx, t_idx]
                    else:
                        y_interp = (1-weight) * y_array[f_idx, t_idx] + weight * y_array[f_idx+1, t_idx]
                    y_arrays_interpolated[j][f_idx, t_idx] = y_interp
        else:
            # Current point is highest, keep original values
            for j, y_array in enumerate(y_arrays):
                y_arrays_interpolated[j][f_idx, t_idx] = y_array[f_idx, t_idx]
    
    return y_arrays_interpolated
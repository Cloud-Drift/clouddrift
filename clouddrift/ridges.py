from numpy.lib import scimath
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from clouddrift.wavelet import (
    morse_wavelet_transform,
    morse_logspace_freq,
)
from typing import Optional, Union, Tuple, List
from scipy import ndimage


def instmom_univariate(
        signal : np.ndarray, 
        sample_rate : float = 1.0, 
        axis : int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate univariate instantaneous moments for a single signal.
    
    This function computes the instantaneous amplitude, frequency, bandwidth, and curvature
    for a single signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal array
    sample_rate : float, optional
        Sample rate for time derivatives, defaults to 1.0
    axis : int, optional
        Axis representing time, defaults to 0

    Returns
    -------
    amplitude : np.ndarray
        Instantaneous amplitude
    omega : np.ndarray
        Instantaneous radian frequency
    upsilon : np.ndarray
        Instantaneous bandwidth
    xi : np.ndarray
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
        signals: np.ndarray,
        sample_rate: float = 1.0,
        time_axis: int = 0,
        joint_axis: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate joint instantaneous moments across multiple signals.
    
    This function computes the joint amplitude, frequency, bandwidth, and curvature
    across multiple signals, with power-weighted averaging of component properties.
    
    Parameters
    ----------
    signals : np.ndarray
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
    joint_amplitude : np.ndarray
        Joint instantaneous amplitude (root-mean-square across signals)
    joint_omega : np.ndarray, optional
        Joint instantaneous radian frequency (power-weighted average)
    joint_upsilon : np.ndarray, optional
        Joint instantaneous bandwidth
    joint_xi : np.ndarray, optional 
        Joint instantaneous curvature
        
    Notes
    -----    
    Reference: Lilly & Olhede (2010)
    """    
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
    wavelet_transform: np.ndarray,
    scale_frequencies: np.ndarray,
    amplitude_threshold: float,
    ridge_type: str,
    freq_min: Optional[Union[float, np.ndarray]] = None,
    freq_max: Optional[Union[float, np.ndarray]] = None,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find wavelet ridge points using specified criterion.
    Ridge detection is performed by finding local maxima along the time axis.
    
    Parameters
    ----------
    wavelet_transform : np.ndarray
        Wavelet transform matrix with shape (time, frequency)
    scale_frequencies : np.ndarray
        Frequencies corresponding to wavelet scales (in radians)
    amplitude_threshold : float
        Minimum amplitude threshold for ridge points
    ridge_type : str
        Ridge definition: 'amplitude' or 'phase'
    freq_min : float or np.ndarray, optional
        Minimum frequency constraint
    freq_max : float or np.ndarray, optional
        Maximum frequency constraint
    mask : np.ndarray, optional
        Boolean mask to restrict ridge locations
    
    Returns
    -------
    ridge_points : np.ndarray
        Boolean matrix indicating ridge points
    ridge_quantity : np.ndarray
        Ridge quantity used for detection
    processed_transform : np.ndarray
        Processed wavelet transform
    inst_frequency : np.ndarray
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

def separate_ridge_groups(
    ridge_points: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Separate ridge points into distinct groups using connected component analysis.
    
    This function identifies separate clusters of ridge points in the time-frequency
    plane. Ridge points are considered connected if they are adjacent according to
    8-connectivity (Moore neighborhood).
    
    Parameters
    ----------
    ridge_points : np.ndarray
        Boolean array where True indicates ridge points, with shape (frequency, time)
        
    Returns
    -------
    labeled_ridges : np.ndarray
        Integer array where values indicate group membership (0 is background)
    num_groups : int
        Number of distinct ridge groups found
        
    Notes
    -----
    This function uses SciPy's ndimage.label to perform connected component analysis.
    """
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_ridges, num_groups = ndimage.label(ridge_points, structure=structure)
    return labeled_ridges, num_groups

def extract_constant_frequency_segments(
    ridge_points: List[Tuple[int, int]],
    freq_tolerance: float = 0.0,
    min_segment_length: int = 2
) -> List[Tuple[int, int]]:
    """
    Find segments of constant frequency and extract central points.
    
    For each segment where the frequency remains constant, identify the point
    closest to the central time of that segment.
    
    Parameters
    ----------
    ridge_points : List[Tuple[int, int]]
        List of (frequency, time) points in a single cluster
    freq_tolerance : float, optional
        Maximum allowed frequency variation to be considered "constant" (default: 0.0)
    min_segment_length : int, optional
        Minimum number of points required for a segment (default: 2)
        
    Returns
    -------
    central_points : List[Tuple[int, int]]
        List of (frequency, time) points representing the central point of each segment
    """
    if not ridge_points:
        return []
    
    # Sort points by time
    ridge_points = sorted(ridge_points, key=lambda pt: pt[1])
    
    central_points = []
    segment_start = 0
    ref_freq = ridge_points[0][0]
    
    # Process points to find segments with constant frequency
    for i in range(1, len(ridge_points)):
        freq, time = ridge_points[i]
        
        # Check if frequency changed beyond tolerance
        if abs(freq - ref_freq) > freq_tolerance:
            # End of segment
            segment_length = i - segment_start
            
            if segment_length >= min_segment_length:
                # Extract segment
                segment = ridge_points[segment_start:i]
                
                # Get times for this segment
                segment_times = [pt[1] for pt in segment]
                
                # Calculate central time
                central_time = sum(segment_times) / len(segment_times)
                
                # Find point closest to central time
                closest_idx = min(range(len(segment)), key=lambda j: abs(segment[j][1] - central_time))
                central_points.append(segment[closest_idx])
            else:
                # For short segments, add all points
                central_points.extend(ridge_points[segment_start:i])
            
            # Start new segment
            segment_start = i
            ref_freq = freq
    
    # Process final segment
    segment_length = len(ridge_points) - segment_start
    if segment_length >= min_segment_length:
        segment = ridge_points[segment_start:]
        segment_times = [pt[1] for pt in segment]
        central_time = sum(segment_times) / len(segment_times)
        closest_idx = min(range(len(segment)), key=lambda j: abs(segment[j][1] - central_time))
        central_points.append(segment[closest_idx])
    else:
        # For short segments, add all points
        central_points.extend(ridge_points[segment_start:])
    
    return central_points
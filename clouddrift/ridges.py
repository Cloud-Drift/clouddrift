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

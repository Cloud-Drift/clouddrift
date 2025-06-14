"""
This module contains functions for converting between float representations of time
"""

import numpy as np


def float_to_datetime64(time_float, unit="s"):
    """
    Convert float seconds (or other units) since UNIX epoch to np.datetime64.

    Parameters:
    ----------
    time_float : float or array-like
        Seconds (or other time units) since epoch (1970-01-01T00:00:00).
    unit : str, optional
        Time unit for conversion. Default is 's' (seconds).
        Valid options: 's', 'ms', 'us', 'ns', etc.

    Returns:
    -------
    np.datetime64 or np.ndarray of np.datetime64
    """
    epoch = np.datetime64("1970-01-01T00:00:00")
    return epoch + time_float.astype(f"timedelta64[{unit}]")


def datetime64_to_float(time_dt, unit="s"):
    """
    Convert np.datetime64 or array of datetime64 to float time since epoch.

    Parameters:
    ----------
    time_dt : np.datetime64 or array-like
        Datetime64 values to convert.
    unit : str, optional
        Unit of the output float (default: 's' for seconds).
        Valid: 's', 'ms', 'us', 'ns', etc.

    Returns:
    -------
    float or np.ndarray of floats
        Time since UNIX epoch (1970-01-01) in specified unit.
    """
    epoch = np.datetime64("1970-01-01T00:00:00")
    return (time_dt - epoch) / np.timedelta64(1, unit)

from datetime import datetime
from typing import Optional, Tuple
import xarray as xr


def velocity_from_positions(
    x: xr.DataArray[float],
    y: xr.DataArray[float],
    time: xr.DataArray[datetime],
    coord_system: Optional[str] = "spherical",
    difference_scheme: Optional[str] = "forward",
) -> Tuple[xr.DataArray[float], xr.DataArray[float]]:
    """Compute velocity in meters per second given arrays of positions and
    time.

    x and y can be provided as longitude and latitude in degrees if
    coord_system == "spherical" (default), or as northing and easting in meters
    if coord_system == "cartesian".

    Difference scheme can take one of three values:

        1. "forward" (default): finite difference is evaluated as dx[i] = dx[i+1] - dx[i];
        2. "backward": finite difference is evaluated as dx[i] = dx[i] - dx[i-1];
        3. "centered": finite difference is evaluated as dx[i] = (dx[i+1] - dx[i-1]) / 2.

    Args:
        x (xr.DataArray[float]): An array of x-positions (longitude in degrees or easting in meters)
        y (xr.DataArray[float]): An array of y-positions (latitude in degrees or northing in meters)
        time (xr.DataArray[datetime]): An array of times
        coord_system (str, optional): Coordinate system that x and y arrays are in; possible values are "spherical" (default) or "cartesian".
        difference_scheme (str, optional): Difference scheme to use; possible values are "forward", "backward", and "centered".

    Returns:
        out (Tuple[xr.DataArray[float], xr.DataArray[float]]): Arrays of x- and y-velocities in meters per second
    """
    raise NotImplementedError()

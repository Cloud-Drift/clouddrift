import numpy as np
from typing import Tuple
import xarray as xr

EARTH_RADIUS_METERS = 6.3781e6


def distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Return elementwise great circle distance in meters between one or more
    points from arrays of their latitudes and longitudes, using the Haversine
    formula.

    d = 2⋅r⋅asin √[sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)]

    where (φ, λ) is (lat, lon) in radians and r is the radius of the sphere in
    meters.

    Args:
        lat1 (array_like): Latitudes of the first set of points, in degrees
        lon1 (array_like): Longitudes of the first set of points, in degrees
        lat2 (array_like): Latitudes of the second set of points, in degrees
        lon2 (array_like): Longitudes of the second set of points, in degrees

    Returns:
        out (array_like): Great circle distance
    """

    # Input coordinates are in degrees; convert to radians.
    # If any of the input arrays are xr.DataArray, extract the values first
    # because Xarray enforces alignment between coordinates.
    if type(lat1) is xr.DataArray:
        lat1_rad = np.deg2rad(lat1.values)
    else:
        lat1_rad = np.deg2rad(lat1)
    if type(lon1) is xr.DataArray:
        lon1_rad = np.deg2rad(lon1.values)
    else:
        lon1_rad = np.deg2rad(lon1)
    if type(lat2) is xr.DataArray:
        lat2_rad = np.deg2rad(lat2.values)
    else:
        lat2_rad = np.deg2rad(lat2)
    if type(lon2) is xr.DataArray:
        lon2_rad = np.deg2rad(lon2.values)
    else:
        lon2_rad = np.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    h = (
        np.sin(0.5 * dlat) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(0.5 * dlon) ** 2
    )

    return 2 * np.arcsin(np.sqrt(h)) * EARTH_RADIUS_METERS


def bearing(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Return elementwise initial (forward) bearing in radians from arrays of
    latitude and longitude in degrees, based on the spherical law of cosines.

    The formula is:

    θ = atan2(cos φ1 ⋅ sin φ2 - sin φ1 ⋅ cos φ2 ⋅ cos Δλ, sin Δλ ⋅ cos φ2)

    where (φ, λ) is (lat, lon) and θ is bearing, all in radians.
    Bearing is defined as zero toward East and positive counterclockwise.

    Args:
        lat1 (array_like): Latitudes of the first set of points, in degrees
        lon1 (array_like): Longitudes of the first set of points, in degrees
        lat2 (array_like): Latitudes of the second set of points, in degrees
        lon2 (array_like): Longitudes of the second set of points, in degrees

    Returns:
        theta (array_like): Bearing angles in radians
    """

    # Input coordinates are in degrees; convert to radians.
    # If any of the input arrays are xr.DataArray, extract the values first
    # because Xarray enforces alignment between coordinates.
    if type(lat1) is xr.DataArray:
        lat1_rad = np.deg2rad(lat1.values)
    else:
        lat1_rad = np.deg2rad(lat1)
    if type(lon1) is xr.DataArray:
        lon1_rad = np.deg2rad(lon1.values)
    else:
        lon1_rad = np.deg2rad(lon1)
    if type(lat2) is xr.DataArray:
        lat2_rad = np.deg2rad(lat2.values)
    else:
        lat2_rad = np.deg2rad(lat2)
    if type(lon2) is xr.DataArray:
        lon2_rad = np.deg2rad(lon2.values)
    else:
        lon2_rad = np.deg2rad(lon2)

    dlon = lon2_rad - lon1_rad

    theta = np.arctan2(
        np.cos(lat1_rad) * np.sin(lat2_rad)
        - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon),
        np.sin(dlon) * np.cos(lat2_rad),
    )

    return theta


def position_from_distance_and_bearing(
    lat: float, lon: float, distance: float, bearing: float
) -> Tuple[float, float]:
    """Return elementwise new position in degrees from arrays of latitude and
    longitude in degrees, distance in meters, and bearing in radians, based on
    the spherical law of cosines.

    The formula is:

    φ2 = asin( sin φ1 ⋅ cos δ + cos φ1 ⋅ sin δ ⋅ cos θ )
    λ2 = λ1 + atan2( sin θ ⋅ sin δ ⋅ cos φ1, cos δ − sin φ1 ⋅ sin φ2 )

    where (φ, λ) is (lat, lon) and θ is bearing, all in radians.
    Bearing is defined as zero toward East and positive counterclockwise.

    Parameters
    ----------
    lat : float
        Latitude of the first set of points, in degrees
    lon : float
        Longitude of the first set of points, in degrees
    distance : array_like
        Distance in meters
    bearing : array_like
        Bearing angles in radians

    Returns
    -------
    lat2 : array_like
        Latitudes of the second set of points, in degrees, in the range [-90, 90]
    lon2 : array_like
        Longitudes of the second set of points, in degrees, in the range [-180, 180]
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    distance_rad = distance / EARTH_RADIUS_METERS

    lat2_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(distance_rad)
        + np.cos(lat_rad) * np.sin(distance_rad) * np.sin(bearing)
    )
    lon2_rad = lon_rad + np.arctan2(
        np.cos(bearing) * np.sin(distance_rad) * np.cos(lat_rad),
        np.cos(distance_rad) - np.sin(lat_rad) * np.sin(lat2_rad),
    )

    return np.rad2deg(lat2_rad), np.rad2deg(lon2_rad)

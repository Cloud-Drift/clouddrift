import numpy as np

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
        lat1 (array_like): Latitudes of the first set of points
        lon1 (array_like): Longitudes of the first set of points
        lat2 (array_like): Latitudes of the second set of points
        lon2 (array_like): Longitudes of the second set of points

    Returns:
        out (array_like): Great circle distance
    """

    # Input coordinates are in degrees; convert to radians.
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
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
    """Return elementwise Haversine initial (forward) bearing in radians from
    arrays of latitude and longitude in degrees.

    The formula is:

    θ = atan2(sin Δλ ⋅ cos φ2, cos φ1 ⋅ sin φ2 - sin φ1 ⋅ cos φ2 ⋅ cos Δλ)

    where (φ, λ) is (lat, lon) and θ is bearing, all in radians.
    Bearing is defined as zero toward North and increasing clockwise.

    Args:
        lat1 (array_like): Latitudes of the first set of points
        lon1 (array_like): Longitudes of the first set of points
        lat2 (array_like): Latitudes of the second set of points
        lon2 (array_like): Longitudes of the second set of points

    Returns:
        theta (array_like): Bearing angles in radians
    """

    # Input coordinates are in degrees; convert to radians.
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    dlon = lon2_rad - lon1_rad

    theta = np.arctan2(
        np.sin(dlon) * np.cos(lat2_rad),
        np.cos(lat1) * np.sin(lat2_rad)
        - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon),
    )

    return theta

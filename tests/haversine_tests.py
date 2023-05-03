from clouddrift.haversine import (
    distance,
    bearing,
    position_from_distance_and_bearing,
    EARTH_RADIUS_METERS,
)
import unittest
import numpy as np


if __name__ == "__main__":
    unittest.main()

one_degree_meters = 2 * np.pi * EARTH_RADIUS_METERS / 360


class haversine_tests(unittest.TestCase):
    def test_distance_one_degree(self):
        self.assertTrue(np.isclose(distance(0, 0, 0, 1), one_degree_meters))
        self.assertTrue(np.isclose(distance(0, 0, 1, 0), one_degree_meters))
        self.assertTrue(np.isclose(distance(0, 0, 0, -1), one_degree_meters))
        self.assertTrue(np.isclose(distance(0, 0, -1, 0), one_degree_meters))

    def test_distance_arraylike(self):
        self.assertTrue(
            np.all(
                np.isclose(
                    distance([0, 0], [0, 0], [0, 0], [1, 1]),
                    np.array(2 * [one_degree_meters]),
                )
            )
        )

    def test_distance_antimeridian(self):
        self.assertTrue(np.isclose(distance(0, 179.5, 0, -179.5), one_degree_meters))
        self.assertTrue(np.isclose(distance(0, -179.5, 0, 179.5), one_degree_meters))
        self.assertTrue(np.isclose(distance(0, 359.5, 0, 360.5), one_degree_meters))
        self.assertTrue(np.isclose(distance(0, 360.5, 0, 359.5), one_degree_meters))

    def test_bearing(self):
        self.assertTrue(np.isclose(bearing(0, 0, 0, 0.1), 0))
        self.assertTrue(np.isclose(bearing(0, 0, 0.1, 0.1), np.pi / 4))
        self.assertTrue(np.isclose(bearing(0, 0, 0.1, 0), np.pi / 2))
        self.assertTrue(np.isclose(bearing(0, 0, 0.1, -0.1), 3 / 4 * np.pi))
        self.assertTrue(np.isclose(bearing(0, 0, 0, -0.1), np.pi))
        self.assertTrue(np.isclose(bearing(0, 0, -0.1, -0.1), -3 / 4 * np.pi))
        self.assertTrue(np.isclose(bearing(0, 0, -0.1, 0), -np.pi / 2))
        self.assertTrue(np.isclose(bearing(0, 0, -0.1, 0.1), -np.pi / 4))

    def test_position_from_distance_and_bearing_one_degree(self):
        self.assertTrue(
            np.allclose(
                position_from_distance_and_bearing(
                    0, 0, np.deg2rad(EARTH_RADIUS_METERS), 0
                ),
                (0, 1),
            )
        )
        self.assertTrue(
            np.allclose(
                position_from_distance_and_bearing(
                    0, 0, np.deg2rad(EARTH_RADIUS_METERS), np.pi / 2
                ),
                (1, 0),
            )
        )
        self.assertTrue(
            np.allclose(
                position_from_distance_and_bearing(
                    0, 0, np.deg2rad(EARTH_RADIUS_METERS), np.pi
                ),
                (0, -1),
            )
        )
        self.assertTrue(
            np.allclose(
                position_from_distance_and_bearing(
                    0, 0, np.deg2rad(EARTH_RADIUS_METERS), 3 * np.pi / 2
                ),
                (-1, 0),
            )
        )

    def test_position_from_distance_and_bearing_antimeridian(self):
        self.assertTrue(
            np.allclose(
                position_from_distance_and_bearing(
                    0, 179.5, np.deg2rad(EARTH_RADIUS_METERS), 0
                ),
                (0, 180.5),
            )
        )

    def test_position_from_distance_and_bearing_roundtrip(self):
        for n in range(100):
            lat1, lon1 = 0, 0
            lat2 = np.random.uniform(-90, 90)
            lon2 = np.random.uniform(-180, 180)
            d = distance(lat1, lon1, lat2, lon2)
            b = bearing(lat1, lon1, lat2, lon2)
            new_lat, new_lon = position_from_distance_and_bearing(lat1, lon1, d, b)
            self.assertTrue(np.allclose((lat2, lon2), (new_lat, new_lon)))

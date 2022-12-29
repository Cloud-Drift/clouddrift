from clouddrift.haversine import distance, bearing, EARTH_RADIUS_METERS
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

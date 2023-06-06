from clouddrift import haversine
from clouddrift.sphere import (
    recast_lon,
    recast_lon180,
    recast_lon360,
    plane_to_sphere,
    sphere_to_plane,
)
import unittest
import numpy as np

ONE_DEGREE_METERS = np.deg2rad(haversine.EARTH_RADIUS_METERS)

if __name__ == "__main__":
    unittest.main()


class recast_longitude_tests(unittest.TestCase):
    def test_same_shape(self):
        self.assertTrue(recast_lon(np.array([200])).shape == np.zeros(1).shape)
        self.assertTrue(recast_lon(np.array([200, 200])).shape == np.zeros(2).shape)
        self.assertIsNone(
            np.testing.assert_equal(
                recast_lon(np.array([[200.5, -200.5], [200.5, -200.5]])).shape,
                np.zeros((2, 2)).shape,
            )
        )

    def test_different_lon0(self):
        self.assertIsNone(
            np.testing.assert_allclose(recast_lon180(np.array([200])), np.array([-160]))
        )
        self.assertIsNone(
            np.testing.assert_allclose(recast_lon360(np.array([200])), np.array([200]))
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([200]), 220), np.array([560])
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([200]), -200), np.array([-160])
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([200]), -200), np.array([-160])
            )
        )

    def test_range_close(self):
        self.assertIsNone(
            np.testing.assert_allclose(recast_lon(np.array([-180])), np.array([-180]))
        )
        self.assertIsNone(
            np.testing.assert_allclose(recast_lon(np.array([0]), 0), np.array([0]))
        )

        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([-20]), -20), np.array([-20])
            )
        )

    def test_multiple_values(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([200, -180, 300])),
                np.array([-160, -180, -60]),
            )
        )

    def test_more_than_360(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([400, -400])), np.array([40, -40])
            )
        )

    def test_decimals(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([200.3, -200.2])), np.array([-159.7, 159.8])
            )
        )


class plane_to_sphere_tests(unittest.TestCase):
    def test_simple(self):
        lon, lat = plane_to_sphere(
            np.array([0.0, ONE_DEGREE_METERS]), np.array([0.0, 0.0])
        )
        self.assertTrue(np.allclose(lon, np.array([0.0, 1.0])))
        self.assertTrue(np.allclose(lat, np.array([0.0, 0.0])))

        lon, lat = plane_to_sphere(
            np.array([0.0, 0.0]), np.array([0.0, ONE_DEGREE_METERS])
        )
        self.assertTrue(np.allclose(lon, np.array([0.0, 0.0])))
        self.assertTrue(np.allclose(lat, np.array([0.0, 1.0])))

    def test_with_origin(self):
        lon_origin = 5
        lat_origin = 0

        lon, lat = plane_to_sphere(
            np.array([0.0, ONE_DEGREE_METERS]),
            np.array([0.0, 0.0]),
            lon_origin,
            lat_origin,
        )
        self.assertTrue(np.allclose(lon, np.array([lon_origin, lon_origin + 1])))
        self.assertTrue(np.allclose(lat, np.array([lat_origin, lat_origin])))

        lon_origin = 0
        lat_origin = 5

        lon, lat = plane_to_sphere(
            np.array([0.0, 0.0]),
            np.array([0.0, ONE_DEGREE_METERS]),
            lon_origin,
            lat_origin,
        )
        self.assertTrue(np.allclose(lon, np.array([lon_origin, lon_origin])))
        self.assertTrue(np.allclose(lat, np.array([lat_origin, lat_origin + 1])))

    def test_scalar_raises_error(self):
        with self.assertRaises(Exception):
            plane_to_sphere(0, 0)


class sphere_to_plane_tests(unittest.TestCase):
    def test_simple(self):
        x, y = sphere_to_plane(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        self.assertTrue(
            np.allclose(x, np.array([0.0, np.deg2rad(haversine.EARTH_RADIUS_METERS)]))
        )
        self.assertTrue(np.allclose(y, np.zeros((2))))

        x, y = sphere_to_plane(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        self.assertTrue(
            np.allclose(y, np.array([0.0, np.deg2rad(haversine.EARTH_RADIUS_METERS)]))
        )
        self.assertTrue(np.allclose(x, np.zeros((2))))

    def test_with_origin(self):
        lon_origin = 5
        lat_origin = 0

        x, y = sphere_to_plane(
            np.array([0.0, 1.0]), np.array([0.0, 0.0]), lon_origin, lat_origin
        )
        self.assertTrue(
            np.allclose(
                x,
                np.array(
                    [
                        0 - lon_origin * ONE_DEGREE_METERS,
                        ONE_DEGREE_METERS - lon_origin * ONE_DEGREE_METERS,
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                y,
                np.array(
                    [-lat_origin * ONE_DEGREE_METERS, -lat_origin * ONE_DEGREE_METERS]
                ),
            )
        )

        lon_origin = 0
        lat_origin = 5

        x, y = sphere_to_plane(
            np.array([0.0, 0.0]), np.array([0.0, 1.0]), lon_origin, lat_origin
        )
        self.assertTrue(
            np.allclose(
                y,
                np.array(
                    [
                        0 - lat_origin * ONE_DEGREE_METERS,
                        ONE_DEGREE_METERS - lat_origin * ONE_DEGREE_METERS,
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                x,
                np.array(
                    [-lon_origin * ONE_DEGREE_METERS, -lon_origin * ONE_DEGREE_METERS]
                ),
            )
        )

    def test_scalar_raises_error(self):
        with self.assertRaises(Exception):
            sphere_to_plane(0, 0)


class sphere_to_plane_roundtrip(unittest.TestCase):
    def test_roundtrip(self):
        expected_lon = 2 * np.cumsum(np.random.random((100)))
        expected_lat = np.cumsum(np.random.random((100)))

        x, y = sphere_to_plane(expected_lon, expected_lat)
        lon, lat = plane_to_sphere(x, y)

        self.assertTrue(np.allclose(lon, expected_lon))
        self.assertTrue(np.allclose(lat, expected_lat))

from clouddrift.sphere import (
    recast_lon,
    recast_lon180,
    recast_lon360,
    plane_to_sphere,
    sphere_to_plane,
    distance,
    bearing,
    position_from_distance_and_bearing,
    spherical_to_cartesian,
    cartesian_to_spherical,
    tangentplane_to_cartesian,
    cartesian_to_tangentplane,
    coriolis_frequency,
    EARTH_RADIUS_METERS,
)
import unittest
import numpy as np

ONE_DEGREE_METERS = np.deg2rad(EARTH_RADIUS_METERS)

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
            np.testing.assert_allclose(recast_lon180(np.array([180])), np.array([-180]))
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
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_lon(np.array([180]), -180), np.array([-180])
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

    def test_distance_one_degree(self):
        self.assertTrue(np.isclose(distance(0, 0, 0, 1), ONE_DEGREE_METERS))
        self.assertTrue(np.isclose(distance(0, 0, 1, 0), ONE_DEGREE_METERS))
        self.assertTrue(np.isclose(distance(0, 0, 0, -1), ONE_DEGREE_METERS))
        self.assertTrue(np.isclose(distance(0, 0, -1, 0), ONE_DEGREE_METERS))

    def test_distance_arraylike(self):
        self.assertTrue(
            np.all(
                np.isclose(
                    distance([0, 0], [0, 0], [0, 0], [1, 1]),
                    np.array(2 * [ONE_DEGREE_METERS]),
                )
            )
        )

    def test_distance_antimeridian(self):
        self.assertTrue(np.isclose(distance(0, 179.5, 0, -179.5), ONE_DEGREE_METERS))
        self.assertTrue(np.isclose(distance(0, -179.5, 0, 179.5), ONE_DEGREE_METERS))
        self.assertTrue(np.isclose(distance(0, 359.5, 0, 360.5), ONE_DEGREE_METERS))
        self.assertTrue(np.isclose(distance(0, 360.5, 0, 359.5), ONE_DEGREE_METERS))

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
            np.allclose(x, np.array([0.0, np.deg2rad(EARTH_RADIUS_METERS)]))
        )
        self.assertTrue(np.allclose(y, np.zeros((2))))

        x, y = sphere_to_plane(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        self.assertTrue(
            np.allclose(y, np.array([0.0, np.deg2rad(EARTH_RADIUS_METERS)]))
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


class spherical_to_cartesian_tests(unittest.TestCase):
    def test_spherical_to_cartesian(self):
        lon = np.array([0, 90, 0, -90, 0]).astype(np.double)
        lat = np.array([0, 0, 45, 45, -90]).astype(np.double)
        x, y, z = spherical_to_cartesian(lon, lat, radius=1)
        x_expected = np.array([1, 0, np.sqrt(2) / 2, 0, 0])
        y_expected = np.array([0, 1, 0, -np.sqrt(2) / 2, 0])
        z_expected = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2, -1])
        self.assertTrue(np.allclose(x, x_expected, atol=1e-6))
        self.assertTrue(np.allclose(y, y_expected, atol=1e-6))
        self.assertTrue(np.allclose(z, z_expected, atol=1e-6))

    def test_spherical_to_cartesian_invert(self):
        lon = np.random.uniform(size=100) * 360
        lat = np.random.uniform(size=100) * 180 - 90
        x, y, z = spherical_to_cartesian(lon, lat)
        lon2, lat2 = cartesian_to_spherical(x, y, z)
        self.assertTrue(np.allclose(lon, np.mod(lon2, 360)))
        self.assertTrue(np.allclose(lat, lat2))


class cartesian_to_spherical_tests(unittest.TestCase):
    def test_cartesian_to_spherical(self):
        x = EARTH_RADIUS_METERS * np.array([1, 0, -1, 0, 0, 0])
        y = EARTH_RADIUS_METERS * np.array([0, 1, 0, -1, 0, 0])
        z = EARTH_RADIUS_METERS * np.array([0, 0, 0, 0, 1, -1])
        lon_expected = np.array([0, 90, -180, -90, 0, 0]).astype("double")
        lat_expected = np.array([0, 0, 0, 0, 90, -90]).astype("double")
        lon, lat = cartesian_to_spherical(x, y, z)
        self.assertTrue(np.allclose(lon, lon_expected))
        self.assertTrue(np.allclose(lat, lat_expected))

    def test_cartesian_to_spherical_invert(self):
        x = np.random.random(size=100)
        y = np.random.random(size=100)
        z = np.random.random(size=100)
        lon, lat = cartesian_to_spherical(x, y, z)
        x2, y2, z2 = spherical_to_cartesian(lon, lat, radius=1)
        self.assertTrue(np.allclose(x, x2))
        self.assertTrue(np.allclose(y, y2))
        self.assertTrue(np.allclose(z, z2))


class cartesian_to_tangentplane_tests(unittest.TestCase):
    def test_cartesian_to_tangentplane_values(self):
        up, vp = cartesian_to_tangentplane(1.0, 1.0, 1.0, 0.0, 0.0)
        self.assertTrue(np.allclose((up, vp), (1.0, 1.0)))
        up, vp = cartesian_to_tangentplane(1.0, 1.0, 1.0, 90.0, 0.0)
        self.assertTrue(np.allclose((up, vp), (-1.0, 1.0)))
        up, vp = cartesian_to_tangentplane(1.0, 1.0, 1.0, 180.0, 0.0)
        self.assertTrue(np.allclose((up, vp), (-1.0, 1.0)))
        up, vp = cartesian_to_tangentplane(1.0, 1.0, 1.0, 270.0, 0.0)
        self.assertTrue(np.allclose((up, vp), (1.0, 1.0)))
        up, vp = cartesian_to_tangentplane(1.0, 1.0, 1.0, 0.0, 90.0)
        self.assertTrue(np.allclose((up, vp), (1.0, -1.0)))
        up, vp = cartesian_to_tangentplane(1.0, 1.0, 1.0, 0.0, -90.0)
        self.assertTrue(np.allclose((up, vp), (1.0, 1.0)))


class tangentplane_to_cartesian_tests(unittest.TestCase):
    def test_tangentplane_to_cartesian_values(self):
        uvw = tangentplane_to_cartesian(1, 1, 0, 0)
        self.assertTrue(np.allclose(uvw, (0, 1, 1)))
        uvw = tangentplane_to_cartesian(1, 1, 90, 0)
        self.assertTrue(np.allclose(uvw, (-1, 0, 1)))
        uvw = tangentplane_to_cartesian(1, 1, 180, 0)
        self.assertTrue(np.allclose(uvw, (0, -1, 1)))
        uvw = tangentplane_to_cartesian(1, 1, 270, 0)
        self.assertTrue(np.allclose(uvw, (1, 0, 1)))
        uvw = tangentplane_to_cartesian(1, 1, 0, 90)
        self.assertTrue(np.allclose(uvw, (-1, 1, 0)))
        uvw = tangentplane_to_cartesian(1, 1, 0, -90)
        self.assertTrue(np.allclose(uvw, (1, 1, 0)))

    def test_tangentplane_to_cartesian_inverse(self):
        u, v, w = tangentplane_to_cartesian(1, 1, 45, 45)
        self.assertTrue(np.allclose((1, 1), cartesian_to_tangentplane(u, v, w, 45, 45)))


class coriolis_frequency_tests(unittest.TestCase):
    def test_coriolis_frequency_values(self):
        f = coriolis_frequency(np.array([-90, -60, -30, 0, 45, 90]))
        f_expected = np.array(
            [
                -0.000145842318,
                -0.000126303152,
                -7.2921159e-05,
                0,
                0.000103126092,
                0.000145842318,
            ]
        )
        self.assertTrue(np.allclose(f, f_expected))

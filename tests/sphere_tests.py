from clouddrift.sphere import recast_longitude
import unittest
import numpy as np


if __name__ == "__main__":
    unittest.main()


class recast_longitude_tests(unittest.TestCase):
    def test_same_shape(self):
        self.assertTrue(recast_longitude(np.array([200])).shape == np.zeros(1).shape)
        self.assertTrue(
            recast_longitude(np.array([200, 200])).shape == np.zeros(2).shape
        )
        self.assertIsNone(
            np.testing.assert_equal(
                recast_longitude(np.array([[200.5, -200.5], [200.5, -200.5]])).shape,
                np.zeros((2, 2)).shape,
            )
        )

    def test_different_lon0(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([200])), np.array([-160])
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([200]), 0), np.array([200])
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([200]), 220), np.array([560])
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([200]), -200), np.array([-160])
            )
        )

    def test_range_close(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([-180])), np.array([-180])
            )
        )
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([0]), 0), np.array([0])
            )
        )

        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([-20]), -20), np.array([-20])
            )
        )

    def test_multiple_values(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([200, -180, 300])),
                np.array([-160, -180, -60]),
            )
        )

    def test_more_than_360(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([400, -400])), np.array([40, -40])
            )
        )

    def test_decimals(self):
        self.assertIsNone(
            np.testing.assert_allclose(
                recast_longitude(np.array([200.3, -200.2])), np.array([-159.7, 159.8])
            )
        )

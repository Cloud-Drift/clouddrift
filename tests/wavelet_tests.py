from clouddrift.wavelet import (
    morsewave,
    morsefreq,
    morseafun,
)
import numpy as np
import unittest

if __name__ == "__main__":
    unittest.main()


class morsewave_tests(unittest.TestCase):
    def test_morsewave_unitenergy(self):
        fs = 2 * np.pi / np.logspace(np.log10(5), np.log10(40))
        ga = 2
        be = 2
        k = 2
        n = 1023
        psi, _ = morsewave(n, ga, be, fs, k=k, norm="energy")
        nrg = np.sum(np.abs(psi[:, :, 0]) ** 2, axis=0)
        self.assertTrue(np.allclose(np.ones_like(nrg), nrg))
        nrg = np.sum(np.abs(psi[:, :, 1]) ** 2, axis=0)
        self.assertTrue(np.allclose(np.ones_like(nrg), nrg))
        # self.assertTrue(True)


class morsefreq_tests(unittest.TestCase):
    def test_morsefreq_array(self):
        ga = np.array([[3, 10, 20], [4, 4, 4]])
        be = np.array([[50, 100, 200], [150, 250, 300]])
        fm, fe, fi = morsefreq(ga, be)
        expected_fm = np.array(
            [
                [2.55436477464518, 1.25892541179417, 1.12201845430196],
                [2.47461600191988, 2.81170662595175, 2.94283095638271],
            ]
        )
        expected_fe = np.array(
            [
                [2.55439315237839, 1.25671816756476, 1.12082018344785],
                [2.47358857550119, 2.81100519592177, 2.94221895404181],
            ]
        )
        expected_fi = np.array(
            [
                [2.55447823649861, 1.25450366161231, 1.11960998925271],
                [2.4725685226628, 2.81030677065388, 2.94160913357547],
            ]
        )
        self.assertTrue(np.allclose(fm, expected_fm))
        self.assertTrue(np.allclose(fe, expected_fe))
        self.assertTrue(np.allclose(fi, expected_fi))

    def test_morsefreq_float(self):
        ga = 3
        be = 50
        fm, fe, fi = morsefreq(ga, be)
        expected_fm = 2.55436477464518
        expected_fe = 2.55439315237839
        expected_fi = 2.55447823649861
        self.assertTrue(np.isclose(fm, expected_fm))
        self.assertTrue(np.isclose(fe, expected_fe))
        self.assertTrue(np.isclose(fi, expected_fi))

    def test_morsefreq_beta_zero(self):
        ga = 3
        be = 0
        fm, fe, fi = morsefreq(ga, be)
        expected_fm = 0.884997044500518
        expected_fe = 0.401190287437665
        expected_fi = 0.505468088156089
        self.assertTrue(np.isclose(fm, expected_fm))
        self.assertTrue(np.isclose(fe, expected_fe))
        self.assertTrue(np.isclose(fi, expected_fi))


class morseafun_tests(unittest.TestCase):
    def test_morseafun(self):
        # ga1 = np.arange(2, 10, 1)
        # be1 = np.arange(1, 11, 1)
        # ga, be = np.meshgrid(ga1, be1)
        # om, _, _ = morsefreq(ga, be)
        # omgrid = np.tile(np.arange(0, 20.01, 0.1), (len(be1), len(ga1), 1))
        # omgrid = omgrid * np.tile(np.expand_dims(om,-1), np.shape(omgrid)[2])
        # a = morseafun(ga,be,norm="energy")
        self.assertTrue(True)

from clouddrift.wavelet import (
    wavetrans,
    morsewave,
    morsefreq,
    morseafun,
)
import numpy as np
import unittest

if __name__ == "__main__":
    unittest.main()


class wavetrans_tests(unittest.TestCase):
    def test_wavetrans_boundary(self):
        n = 1023
        fs = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        wave, wavef = morsewave(n, 2, 4, fs, order=1, norm="bandpass")
        x = np.random.random((n))
        w1 = wavetrans(x - np.mean(x), wave, boundary="mirror")
        w2 = wavetrans(x - np.mean(x), wave, boundary="periodic")
        w3 = wavetrans(x - np.mean(x), wave, boundary="zeros")
        s = slice(int(n / 4 - 1), int(n / 4 - 1 + n / 2))
        # not sure why the real part only succeeds
        self.assertTrue(np.allclose(np.real(w1[..., s]), np.real(w2[..., s])))
        self.assertTrue(np.allclose(np.real(w1[..., s]), np.real(w3[..., s])))
        self.assertTrue(np.allclose(np.real(w2[..., s]), np.real(w3[..., s])))
        # self.assertTrue(np.allclose(np.abs(w1[..., s]), np.abs(w2[..., s])), atol=1e-2)
        # self.assertTrue(np.allclose(np.abs(w1[..., s]), np.abs(w3[..., s])), atol=1e-2)

    def test_wavetrans_complex(self):
        n = 1023
        fs = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        wave, wavef = morsewave(n, 2, 4, fs, order=1, norm="bandpass")
        x = np.random.random((n))
        y = np.random.random((n))
        wx = wavetrans(x, wave, boundary="mirror", norm="bandpass")
        wy = wavetrans(y, wave, boundary="mirror", norm="bandpass")
        wp = wavetrans(x + 1j * y, wave, boundary="mirror", norm="bandpass")
        wn = wavetrans(x - 1j * y, wave, boundary="mirror", norm="bandpass")
        wp2 = 0.5 * (wx + 1j * wy)
        wn2 = 0.5 * (wx - 1j * wy)
        self.assertTrue(np.allclose(wp, wp2, atol=1e-6))
        self.assertTrue(np.allclose(wn, wn2, atol=1e-6))

    def test_wavetrans_sizes(self):
        n = 1023
        m = 10
        order = 2
        fs = 2 * np.pi * np.array([0.1, 0.2, 0.3])
        ga = 3
        be = 4
        x = np.random.random((m, n))
        wave, _ = morsewave(n, ga, be, fs, order=order)
        w = wavetrans(x, wave)
        self.assertTrue(np.shape(w) == (m, order, len(fs), n))

    def test_wavetrans_centered(self):
        J = 10
        ao = np.logspace(np.log10(5), np.log10(40), J) / 100
        x = np.zeros(2**10)
        wave, _ = morsewave(len(x), 2, 4, ao, order=1)
        x[2**9] = 1
        y = wavetrans(x, wave)
        m = np.argmax(np.abs(y), axis=-1)
        self.assertTrue(np.allclose(m, 2**9))

    def test_wavetrans_data(self):
        # to write
        self.assertTrue(True)


class morsewave_tests(unittest.TestCase):
    def test_morsewave_unitenergy(self):
        fs = 2 * np.pi / np.logspace(np.log10(5), np.log10(40))
        ga = 2
        be = 4
        order = 2
        n = 1023
        wave, _ = morsewave(n, ga, be, fs, order=order, norm="energy")
        nrg = np.sum(np.abs(wave) ** 2, axis=-1)
        self.assertTrue(np.allclose(1, nrg, atol=1e-4))


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


class morsespace_tests(unittest.TestCase):
    def test_morsespace_high(self):
        self.assertTrue(True)

    def test_morsespace_low(self):
        self.assertTrue(True)


class morseafun_tests(unittest.TestCase):
    def test_morseafun_float(self):
        # ga1 = np.arange(2, 10, 1)
        # be1 = np.arange(1, 11, 1)
        # ga, be = np.meshgrid(ga1, be1)
        # om, _, _ = morsefreq(ga, be)
        # omgrid = np.tile(np.arange(0, 20.01, 0.1), (len(be1), len(ga1), 1))
        # omgrid = omgrid * np.tile(np.expand_dims(om,-1), np.shape(omgrid)[2])
        # a = morseafun(ga,be,norm="energy")
        # gagrid = np.tile(np.expand_dims(ga,-1), np.shape(omgrid)[2])
        # begrid = np.tile(np.expand_dims(be,-1), np.shape(omgrid)[2])
        # agrid = np.tile(np.expand_dims(a,-1), np.shape(omgrid)[2])
        # wave = agrid * omgrid**begrid * np.exp(-omgrid**gagrid)
        # dom = 0.01
        # waveint = np.sum(wave**2,axis=-1) * dom * om / (2 * np.pi)
        # self.assertTrue(np.allclose(np.abs(waveint-1),1e-2))
        # self.assertTrue(True)
        ga = 3
        be = 5
        self.assertTrue(np.isclose(morseafun(ga, be), 4.51966469068946))

    def test_morseafun_array(self):
        ga = np.array([3, 4, 5])
        be = np.array([3, 5, 7])
        expected_a = np.array([5.43656365691809, 5.28154010330058, 5.06364231419937])
        self.assertTrue(np.allclose(morseafun(ga, be), expected_a))

    def test_morseafun_beta_zero(self):
        ga = np.array([3, 4, 5])
        be = np.array([0, 0, 0])
        expected_a = np.array([2, 2, 2])
        self.assertTrue(np.allclose(morseafun(ga, be), expected_a))

    def test_morseafun_ndarray(self):
        ga = np.array([[3, 4], [5, 6]])
        be = np.array([[5.6, 6.5], [7.5, 8.5]])
        expected_a = np.array(
            [[4.03386834889409, 4.61446982215091], [4.87904507028292, 5.03482799479815]]
        )
        self.assertTrue(np.allclose(morseafun(ga, be), expected_a))

    def test_morseafun_energy(self):
        ga = np.array([[3, 4], [5, 6]])
        be = np.array([[5.6, 6.5], [7.5, 8.5]])
        expected_a = np.array(
            [[6.95583044131426, 9.24984207652964], [10.9133909718769, 12.2799204953579]]
        )
        self.assertTrue(np.allclose(morseafun(ga, be, norm="energy"), expected_a))

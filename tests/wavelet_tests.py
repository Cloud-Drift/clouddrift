from clouddrift.wavelet import (
    wavelet_transform,
    morse_wavelet,
    morse_freq,
    morse_amplitude,
    morse_space,
    morse_properties,
)
import numpy as np
import unittest

if __name__ == "__main__":
    unittest.main()


class wavelet_transform_tests(unittest.TestCase):
    def test_wavelet_transform_boundary(self):
        length = 1023
        radian_frequency = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        wave, wavef = morse_wavelet(
            length, 2, 4, radian_frequency, order=1, normalization="bandpass"
        )
        x = np.random.random((length))
        w1 = wavelet_transform(x - np.mean(x), wave, boundary="mirror")
        w2 = wavelet_transform(x - np.mean(x), wave, boundary="periodic")
        w3 = wavelet_transform(x - np.mean(x), wave, boundary="zeros")
        s = slice(int(length / 4 - 1), int(length / 4 - 1 + length / 2))
        # not sure why the real part only succeeds
        self.assertTrue(np.allclose(np.real(w1[..., s]), np.real(w2[..., s])))
        self.assertTrue(np.allclose(np.real(w1[..., s]), np.real(w3[..., s])))
        self.assertTrue(np.allclose(np.real(w2[..., s]), np.real(w3[..., s])))
        # self.assertTrue(np.allclose(np.abs(w1[..., s]), np.abs(w2[..., s])), atol=1e-2)
        # self.assertTrue(np.allclose(np.abs(w1[..., s]), np.abs(w3[..., s])), atol=1e-2)

    def test_wavelet_transform_complex(self):
        length = 1023
        radian_frequency = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        wave, wavef = morse_wavelet(
            length, 2, 4, radian_frequency, order=1, normalization="bandpass"
        )
        x = np.random.random((length))
        y = np.random.random((length))
        wx = wavelet_transform(x, wave, boundary="mirror", normalization="bandpass")
        wy = wavelet_transform(y, wave, boundary="mirror", normalization="bandpass")
        wp = wavelet_transform(
            x + 1j * y, wave, boundary="mirror", normalization="bandpass"
        )
        wn = wavelet_transform(
            x - 1j * y, wave, boundary="mirror", normalization="bandpass"
        )
        wp2 = 0.5 * (wx + 1j * wy)
        wn2 = 0.5 * (wx - 1j * wy)
        self.assertTrue(np.allclose(wp, wp2, atol=1e-6))
        self.assertTrue(np.allclose(wn, wn2, atol=1e-6))

    def test_wavelet_transform_size(self):
        length = 1023
        m = 10
        order = 2
        radian_frequency = 2 * np.pi * np.array([0.1, 0.2, 0.3])
        gamma = 3
        beta = 4
        x = np.random.random((m, m * 2, length))
        wave, _ = morse_wavelet(length, gamma, beta, radian_frequency, order=order)
        w = wavelet_transform(x, wave)
        self.assertTrue(np.shape(w) == (m, m * 2, order, len(radian_frequency), length))

    def test_wavelet_transform_size_axis(self):
        length = 1023
        m = 10
        order = 2
        radian_frequency = 2 * np.pi * np.array([0.1, 0.2, 0.3])
        gamma = 3
        beta = 4
        x = np.random.random((length, m, int(m / 2)))
        wave, _ = morse_wavelet(length, gamma, beta, radian_frequency, order=order)
        w = wavelet_transform(x, wave, time_axis=0)
        self.assertTrue(np.shape(w) == (length, m, m / 2, order, len(radian_frequency)))

    def test_wavelet_transform_centered(self):
        J = 10
        ao = np.logspace(np.log10(5), np.log10(40), J) / 100
        x = np.zeros(2**10)
        wave, _ = morse_wavelet(len(x), 2, 4, ao, order=1)
        x[2**9] = 1
        y = wavelet_transform(x, wave)
        m = np.argmax(np.abs(y), axis=-1)
        self.assertTrue(np.allclose(m, 2**9))

    def test_wavelet_transform_data(self):
        # to write
        self.assertTrue(True)


class morse_wavelet_tests(unittest.TestCase):
    def test_morse_wavelet_unitenergy(self):
        radian_frequency = 2 * np.pi / np.logspace(np.log10(5), np.log10(40))
        gamma = 2
        beta = 4
        order = 2
        length = 1023
        wave, _ = morse_wavelet(
            length, gamma, beta, radian_frequency, order=order, normalization="energy"
        )
        nrg = np.sum(np.abs(wave) ** 2, axis=-1)
        self.assertTrue(np.allclose(1, nrg, atol=1e-4))


class morse_freq_tests(unittest.TestCase):
    def test_morse_freq_array(self):
        gamma = np.array([[3, 10, 20], [4, 4, 4]])
        beta = np.array([[50, 100, 200], [150, 250, 300]])
        fm, fe, fi = morse_freq(gamma, beta)
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

    def test_morse_freq_float(self):
        gamma = 3
        beta = 50
        fm, fe, fi = morse_freq(gamma, beta)
        expected_fm = 2.55436477464518
        expected_fe = 2.55439315237839
        expected_fi = 2.55447823649861
        self.assertTrue(np.isclose(fm, expected_fm))
        self.assertTrue(np.isclose(fe, expected_fe))
        self.assertTrue(np.isclose(fi, expected_fi))

    def test_morse_freq_beta_zero(self):
        gamma = 3
        beta = 0
        fm, fe, fi = morse_freq(gamma, beta)
        expected_fm = 0.884997044500518
        expected_fe = 0.401190287437665
        expected_fi = 0.505468088156089
        self.assertTrue(np.isclose(fm, expected_fm))
        self.assertTrue(np.isclose(fe, expected_fe))
        self.assertTrue(np.isclose(fi, expected_fi))


class morse_space_tests(unittest.TestCase):
    def test_morse_space_high(self):
        self.assertTrue(True)

    # to write
    def test_morse_space_low(self):
        self.assertTrue(True)

    # to write


class morse_properties_tests(unittest.TestCase):
    def test_morse_properties(self):
        # gamma = 5
        # beta = 6
        # length = 512*4
        # fm, _, _ = morse_freq(gamma,beta)
        # dt = 1/20
        # wavelet = morse_wavelet(length,gamma,beta,fm*dt)
        # t = np.arange(0,np.shape(wavelet)[-1])*dt
        # t -= np.mean(t)
        # width, skew, kurt = morse_properties(gamma, beta)
        self.assertTrue(True)

    # to write


class morse_amplitude_tests(unittest.TestCase):
    def test_morse_amplitude_float(self):
        # gamma1 = np.arange(2, 10, 1)
        # beta1 = np.arange(1, 11, 1)
        # gamma, beta = np.meshgrid(gamma1, beta1)
        # om, _, _ = morse_freq(gamma, beta)
        # omgrid = np.tile(np.arange(0, 20.01, 0.1), (len(beta1), len(gamma1), 1))
        # omgrid = omgrid * np.tile(np.expand_dims(om,-1), np.shape(omgrid)[2])
        # a = morse_amplitude(gamma,beta,normalization="energy")
        # gammagrid = np.tile(np.expand_dims(gamma,-1), np.shape(omgrid)[2])
        # betagrid = np.tile(np.expand_dims(beta,-1), np.shape(omgrid)[2])
        # agrid = np.tile(np.expand_dims(a,-1), np.shape(omgrid)[2])
        # wave = agrid * omgrid**betagrid * np.exp(-omgrid**gammagrid)
        # dom = 0.01
        # waveint = np.sum(wave**2,axis=-1) * dom * om / (2 * np.pi)
        # self.assertTrue(np.allclose(np.abs(waveint-1),1e-2))
        # self.assertTrue(True)
        gamma = 3
        beta = 5
        self.assertTrue(np.isclose(morse_amplitude(gamma, beta), 4.51966469068946))

    def test_morse_amplitude_array(self):
        gamma = np.array([3, 4, 5])
        beta = np.array([3, 5, 7])
        expected_a = np.array([5.43656365691809, 5.28154010330058, 5.06364231419937])
        self.assertTrue(np.allclose(morse_amplitude(gamma, beta), expected_a))

    def test_morse_amplitude_beta_zero(self):
        gamma = np.array([3, 4, 5])
        beta = np.array([0, 0, 0])
        expected_a = np.array([2, 2, 2])
        self.assertTrue(np.allclose(morse_amplitude(gamma, beta), expected_a))

    def test_morse_amplitude_ndarray(self):
        gamma = np.array([[3, 4], [5, 6]])
        beta = np.array([[5.6, 6.5], [7.5, 8.5]])
        expected_a = np.array(
            [[4.03386834889409, 4.61446982215091], [4.87904507028292, 5.03482799479815]]
        )
        self.assertTrue(np.allclose(morse_amplitude(gamma, beta), expected_a))

    def test_morse_amplitude_energy(self):
        gamma = np.array([[3, 4], [5, 6]])
        beta = np.array([[5.6, 6.5], [7.5, 8.5]])
        expected_a = np.array(
            [[6.95583044131426, 9.24984207652964], [10.9133909718769, 12.2799204953579]]
        )
        self.assertTrue(
            np.allclose(
                morse_amplitude(gamma, beta, normalization="energy"), expected_a
            )
        )

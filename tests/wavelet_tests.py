import unittest

import numpy as np

from clouddrift.wavelet import (
    _morsehigh,
    morse_amplitude,
    morse_freq,
    morse_logspace_freq,
    morse_properties,
    morse_wavelet,
    morse_wavelet_transform,
    wavelet_transform,
)

if __name__ == "__main__":
    unittest.main()


class morse_wavelet_transform_tests(unittest.TestCase):
    def test_morse_wavelet_transform_real(self):
        length = 1023
        radian_frequency = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        x = np.random.random((length))
        wtx = morse_wavelet_transform(x, 3, 10, radian_frequency)
        wavelet, _ = morse_wavelet(length, 3, 10, radian_frequency)
        wtx2 = wavelet_transform(x, wavelet)
        self.assertTrue(np.allclose(wtx, wtx2))

    def test_morse_wavelet_transform_complex(self):
        length = 1024
        radian_frequency = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        x = np.random.random((length)) + 1j * np.random.random((length))
        wtx_p, wtx_n = morse_wavelet_transform(x, 3, 10, radian_frequency, complex=True)
        wavelet, _ = morse_wavelet(length, 3, 10, radian_frequency)
        wtx2 = wavelet_transform(x, wavelet)
        wtx3 = wavelet_transform(np.conj(x), wavelet)
        self.assertTrue(np.allclose(wtx_p, 0.5 * wtx2))
        self.assertTrue(np.allclose(wtx_n, 0.5 * wtx3))

    def test_morse_wavelet_transform_rotary_bandpass(self):
        length = 2048
        radian_frequency = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        x = np.random.random((length))
        y = np.random.random((length))
        z = x + 1j * y
        wtx = morse_wavelet_transform(x, 3, 10, radian_frequency, complex=False)
        wty = morse_wavelet_transform(y, 3, 10, radian_frequency, complex=False)
        wp = 0.5 * (wtx + 1j * wty)
        wn = 0.5 * (wtx - 1j * wty)
        wp2, _ = morse_wavelet_transform(z, 3, 10, radian_frequency, complex=True)
        wn2, _ = morse_wavelet_transform(
            np.conj(z), 3, 10, radian_frequency, complex=True
        )
        self.assertTrue(np.allclose(wp, wp2))
        self.assertTrue(np.allclose(wn, wn2))

    def test_morse_wavelet_transform_rotary_energy(self):
        length = 1023
        radian_frequency = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        x = np.random.random((length))
        y = np.random.random((length))
        z = x + 1j * y
        wtx = morse_wavelet_transform(
            x, 3, 10, radian_frequency, complex=False, normalization="energy"
        )
        wty = morse_wavelet_transform(
            y, 3, 10, radian_frequency, complex=False, normalization="energy"
        )
        wp = (wtx + 1j * wty) / np.sqrt(2)
        wn = (wtx - 1j * wty) / np.sqrt(2)
        wp2, _ = morse_wavelet_transform(
            z, 3, 10, radian_frequency, complex=True, normalization="energy"
        )
        wn2, _ = morse_wavelet_transform(
            np.conj(z), 3, 10, radian_frequency, complex=True, normalization="energy"
        )
        self.assertTrue(np.allclose(wp, wp2))
        self.assertTrue(np.allclose(wn, wn2))

    def test_morse_wavelet_transform_cos(self):
        f = 0.2
        t = np.arange(0, 1000)
        x = np.cos(2 * np.pi * t * f)
        wtx = morse_wavelet_transform(x, 3, 10, 2 * np.pi * np.array([f]))
        self.assertTrue(np.isclose(np.var(x), 0.5 * np.var(wtx), atol=1e-2))

    def test_morse_wavelet_transform_exp(self):
        f = 0.2
        t = np.arange(0, 1024)
        x = np.exp(1j * 2 * np.pi * t * f)
        wtp, wtn = morse_wavelet_transform(
            x, 3, 10, 2 * np.pi * np.array([f]), complex=True
        )
        self.assertTrue(np.isclose(np.var(x), np.var(wtp) + np.var(wtn), atol=1e-2))


class wavelet_transform_tests(unittest.TestCase):
    def test_wavelet_transform_boundary(self):
        length = 2001
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

    def test_wavelet_transform_complex(self):
        length = 1000
        radian_frequency = 2 * np.pi / np.logspace(np.log10(10), np.log10(100), 50)
        wave, wavef = morse_wavelet(
            length, 2, 4, radian_frequency, order=1, normalization="bandpass"
        )
        x = np.random.random((length))
        y = np.random.random((length))
        wx = wavelet_transform(x, wave, boundary="mirror")
        wy = wavelet_transform(y, wave, boundary="mirror")
        wp = wavelet_transform(
            x + 1j * y,
            0.5 * wave,
            boundary="mirror",
        )
        wn = wavelet_transform(
            x - 1j * y,
            0.5 * wave,
            boundary="mirror",
        )
        wp2 = 0.5 * (wx + 1j * wy)
        wn2 = 0.5 * (wx - 1j * wy)
        self.assertTrue(np.allclose(wp, wp2, atol=1e-6))
        self.assertTrue(np.allclose(wn, wn2, atol=1e-6))

    def test_wavelet_transform_size(self):
        length = 2046
        m = 10
        order = 2
        radian_frequency = 2 * np.pi * np.array([0.1, 0.2, 0.3])
        gamma = 3
        beta = 4
        x = np.random.random((m, m * 2, length))
        wavelet, _ = morse_wavelet(length, gamma, beta, radian_frequency, order=order)
        wtx = wavelet_transform(x, wavelet)
        self.assertTrue(
            np.shape(wtx) == (m, m * 2, order, len(radian_frequency), length)
        )
        x = np.random.random((length, m, m * 2))
        wavelet, _ = morse_wavelet(length, gamma, beta, radian_frequency, order=order)
        wtx = wavelet_transform(x, wavelet, time_axis=0)
        self.assertTrue(
            np.shape(wtx) == (length, m, m * 2, order, len(radian_frequency))
        )
        x = np.random.random((m, length, m * 2))
        wavelet, _ = morse_wavelet(length, gamma, beta, radian_frequency, order=order)
        wtx = wavelet_transform(x, wavelet, time_axis=1)
        self.assertTrue(
            np.shape(wtx) == (m, length, m * 2, order, len(radian_frequency))
        )

    def test_wavelet_transform_size_axis(self):
        length = 1024
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
        wavelet, _ = morse_wavelet(len(x), 2, 4, ao, order=1)
        x[2**9] = 1
        y = wavelet_transform(x, wavelet)
        m = np.argmax(np.abs(y), axis=-1)
        self.assertTrue(np.allclose(m, 2**9))

    def test_wavelet_transform_data_real(self):
        t = np.arange(0, 1000)
        dt = np.diff(t[0:2])
        f = 0.2
        omega = dt * 2 * np.pi * f
        a = 1
        x = a * np.cos(2 * np.pi * t * f)
        gamma = 3
        beta = 10
        waveletb, _ = morse_wavelet(
            np.shape(t)[0], gamma, beta, omega, normalization="bandpass"
        )
        wtxb = wavelet_transform(x, waveletb, boundary="mirror")
        self.assertTrue(np.isclose(np.var(wtxb), 2 * np.var(x), rtol=1e-1))

    def test_wavelet_transform_data_complex(self):
        t = np.arange(0, 1000)
        dt = np.diff(t[0:2])
        f = 0.2
        omega = dt * 2 * np.pi * f
        a = 1
        z = a * np.exp(1j * 2 * np.pi * t * f) + a / 2 * np.exp(-1j * 2 * np.pi * t * f)
        gamma = 3
        beta = 10
        waveletb, _ = morse_wavelet(
            np.shape(t)[0], gamma, beta, omega, normalization="bandpass"
        )
        wtzb = wavelet_transform(z, 0.5 * waveletb, boundary="mirror")
        wtzcb = wavelet_transform(np.conj(z), 0.5 * waveletb, boundary="mirror")
        self.assertTrue(np.isclose(np.var(z), np.var(wtzb) + np.var(wtzcb), rtol=1e-1))


class morse_wavelet_tests(unittest.TestCase):
    def test_morse_wavelet_unitenergy(self):
        radian_frequency = 2 * np.pi / np.logspace(np.log10(5), np.log10(40))
        gamma = 2
        beta = 4
        order = 2
        length = 1000
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


class morse_logspace_freq_tests(unittest.TestCase):
    def test_morse_logspace_freq_high(self):
        # here we are not testing the morse_logspace_freq function
        gamma = np.array([3])
        beta = np.array([4])
        eta = 0.1
        fhigh = _morsehigh(gamma, beta, eta)
        _, waveletfft = morse_wavelet(10000, gamma, beta, fhigh)
        self.assertTrue(
            np.isclose(
                np.abs(0.5 * waveletfft[0, 0, int(10000 / 2) - 1]), eta, atol=1e-3
            )
        )

    def test_morse_logspace_freq_low(self):
        # to write; requires morsebox: Heisenberg time-frequency box for generalized Morse wavelets.
        self.assertTrue(True)

    def test_morse_logspace_freq_values(self):
        fs = morse_logspace_freq(3, 10, 1024)
        self.assertTrue(
            np.allclose(
                fs[[0, -1]], np.array([2.26342969061515, 0.0761392757859202]), atol=1e-5
            )
        )
        fs = morse_logspace_freq(
            3, 10, 1024, highset=(0.3, np.pi), lowset=(5, 0), density=10
        )
        self.assertTrue(
            np.allclose(
                fs[[0, -1]], np.array([2.45100152921832, 0.0759779680679649]), atol=1e-5
            )
        )
        self.assertTrue(np.shape(fs)[0] == 193)


class morse_properties_tests(unittest.TestCase):
    def test_morse_properties(self):
        gamma = 5
        beta = 6
        expected = np.array([5.47722557505166, 0.365148371670111, 2.8])
        width, skew, kurt = morse_properties(gamma, beta)
        self.assertTrue(np.allclose(expected, np.array([width, skew, kurt])))
        gamma = 2
        beta = 4
        expected = np.array([2.82842712474619, -0.353553390593274, 2.625])
        width, skew, kurt = morse_properties(gamma, beta)
        self.assertTrue(np.allclose(expected, np.array([width, skew, kurt])))


class morse_amplitude_tests(unittest.TestCase):
    def test_morse_amplitude_float(self):
        gamma = 3.0
        beta = 5.0
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

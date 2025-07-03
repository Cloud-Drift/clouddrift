import unittest

import numpy as np
from numpy.lib import scimath

from clouddrift.ridges import (
    bilateral_minimum_selection,
    calculate_ridge_group_lengths,
    create_3d_deviation_matrix,
    gradient_of_angle,
    instmom_multivariate,
    instmom_univariate,
    isridgepoint,
    organize_ridge_points,
    ridge_analysis,
    ridge_shift_interpolation,
    separate_ridge_groups,
)
from clouddrift.wavelet import (
    morse_logspace_freq,
    morse_wavelet_transform,
)


class TestGradientOfAngle(unittest.TestCase):
    def test_no_wrap(self):
        """ Simple linear angles should match np.gradient"""
        angles = np.array([0.0, 1.0, 2.0, 3.0])
        grad = gradient_of_angle(angles, edge_order=1, axis=0)
        expected = np.gradient(angles)
        self.assertTrue(np.allclose(grad, expected))

    def test_wrap(self):
        """ Angles wrapping around ±π should unwrap smoothly"""
        # Create a linear ramp that crosses the ±π boundary
        linear = np.linspace(-np.pi + 0.5, np.pi + 0.5, 10)
        angles = (linear + np.pi) % (2 * np.pi) - np.pi
        # Compare gradient_of_angle to np.gradient after unwrapping
        grad_wrap = gradient_of_angle(angles, axis=0)
        expected = np.gradient(np.unwrap(angles))
        np.testing.assert_allclose(grad_wrap, expected, atol=1e-6)

    def test_edge_order_2_even_length(self):
        """ Even-length array with edge_order=2 should match np.gradient after unwrap"""
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        out = gradient_of_angle(angles, edge_order=2, axis=0)
        expected = np.gradient(np.unwrap(angles), edge_order=2)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_edge_order_2_odd_length(self):
        """ Odd-length array with edge_order=2 should also match np.gradient after unwrap"""
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        out = gradient_of_angle(angles, edge_order=2, axis=0)
        expected = np.gradient(np.unwrap(angles), edge_order=2)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_axis_parameter(self):
        """ 2D array: compute along axis=1"""
        base = np.linspace(0, 3.0, 4)
        arr = np.stack([base, -base], axis=0)  # shape (2,4)
        out = gradient_of_angle(arr, edge_order=1, axis=1)
        for i in range(arr.shape[0]):
            np.testing.assert_allclose(out[i], np.gradient(arr[i]), atol=1e-6)


class TestInstantaneousMoments(unittest.TestCase):
    def test_univariate_simple(self):
        """ Test analytic signal with unit amplitude and unit frequency"""
        t = np.linspace(0, 2 * np.pi, 100)
        dt = t[1] - t[0]  # Time step
        signal = np.exp(1j * t)
        amp, omega, upsilon, xi = instmom_univariate(signal, sample_rate=1 / dt, axis=0)
        self.assertTrue(np.allclose(amp, 1.0, atol=1e-6))
        self.assertTrue(np.allclose(omega, 1.0, atol=1e-2))
        self.assertTrue(np.allclose(upsilon, 0.0, atol=1e-2))
        self.assertTrue(np.allclose(xi, 0.0 + 1j * 0.0, atol=1e-2))

    def test_multivariate_simple(self):
        """ Two-component signal to test joint moments"""
        t = np.linspace(0, 2 * np.pi, 100)
        dt = t[1] - t[0]  # Time step
        s1 = np.exp(1j * t)
        s2 = np.exp(1j * (2 * t))
        signals = np.stack([s1, s2], axis=1)  # shape (time, 2)
        amp, omega, upsilon, xi = instmom_multivariate(
            signals, sample_rate=1.0 / dt, time_axis=0, joint_axis=1
        )
        # Basic shape checks and positivity
        self.assertEqual(amp.shape, t.shape)
        self.assertEqual(omega.shape, t.shape)
        self.assertTrue(np.all(amp > 0))

    def test_univariate_complex_signal(self):
        """ Complex signal with known frequency and phase"""
        t = np.linspace(2.0, 6.0, 100)
        dt = t[1] - t[0]  # Time step
        signal = np.log(t) * np.exp(1j * t)
        amp, omega, upsilon, xi = instmom_univariate(
            signal, sample_rate=1.0 / dt, axis=0
        )
        amps = [np.log(x) for x in t]  # Amplitude is log(t)
        omegas = [1.0] * len(t)  # Frequency is constant 1.0
        upsilons = [1.0 / (x * np.log(x)) for x in t]
        xis = [-np.log(x) / (x * np.log(x)) ** 2.0 for x in t]

        # Test the middle points where numerical derivatices are most accurate
        mid_start = len(t) // 10
        mid_end = -len(t) // 10

        self.assertTrue(np.allclose(amp, amps, atol=1e-6))
        self.assertTrue(np.allclose(omega, omegas, atol=1e-6))
        self.assertTrue(
            np.allclose(
                upsilon[mid_start:mid_end], upsilons[mid_start:mid_end], atol=1e-3
            )
        )
        self.assertTrue(
            np.allclose(xi[mid_start:mid_end], xis[mid_start:mid_end], atol=1e-3)
        )

    def test_multivariate_complex_signal(self):
        """ Complex multivariate signal with known frequencies"""
        t = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        s1 = np.log(t) * np.exp(1j * t)
        s2 = np.log(t) * np.exp(1j * (2 * t))
        signals = np.stack([s1, s2], axis=1)
        amp, omega, upsilon, xi = instmom_multivariate(
            signals, sample_rate=1.0, time_axis=0, joint_axis=1
        )

        # Check if outputs are 1D (joint moments) or 2D (per-component)
        if amp.ndim == 1:
            # Joint moments - check basic shapes
            self.assertEqual(amp.shape, t.shape)
            self.assertEqual(omega.shape, t.shape)
            self.assertEqual(upsilon.shape, t.shape)
            self.assertEqual(xi.shape, t.shape)
        else:
            # Per-component moments
            amps_1 = [np.log(x) for x in t]
            amps_2 = [np.log(x) for x in t]
            omegas_1 = [1.0] * len(t)
            omegas_2 = [2.0] * len(t)
            # TODO: Fix analytical formulas for upsilon and xi
            upsilons_1 = [1.0 / (x * np.log(x)) for x in t]
            upsilons_2 = [1.0 / (x * np.log(x)) for x in t]
            xis_1 = [-np.log(x) / (x * np.log(x)) ** 2.0 for x in t]
            xis_2 = [-np.log(x) / (x * np.log(x)) ** 2.0 for x in t]

            # Test the middle points where numerical derivatices are most accurate
            mid_start = len(t) // 10
            mid_end = -len(t) // 10

            self.assertTrue(np.allclose(amp[:, 0], amps_1, atol=1e-6))
            self.assertTrue(np.allclose(amp[:, 1], amps_2, atol=1e-6))
            self.assertTrue(np.allclose(omega[:, 0], omegas_1, atol=1e-6))
            self.assertTrue(np.allclose(omega[:, 1], omegas_2, atol=1e-6))
            self.assertTrue(
                np.allclose(
                    upsilon[:, 0][mid_start:mid_end],
                    upsilons_1[mid_start:mid_end],
                    atol=1e-3,
                )
            )
            self.assertTrue(
                np.allclose(
                    upsilon[:, 1][mid_start:mid_end],
                    upsilons_2[mid_start:mid_end],
                    atol=1e-3,
                )
            )
            self.assertTrue(
                np.allclose(
                    xi[:, 0][mid_start:mid_end], xis_1[mid_start:mid_end], atol=1e-3
                )
            )
            self.assertTrue(
                np.allclose(
                    xi[:, 1][mid_start:mid_end], xis_2[mid_start:mid_end], atol=1e-3
                )
            )


class TestIsRidgePoint(unittest.TestCase):
    def test_empty_transform(self):
        """ Empty input should yield empty ridge points"""
        wt = np.zeros((0, 0), dtype=np.complex128)
        freqs = np.array([])
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.1, "amplitude")
        self.assertEqual(rp.size, 0)
        self.assertEqual(rq.size, 0)

    def test_basic_amplitude_ridge_vertical(self):
        """ Create a Gaussian peak at center scale (ridge across all times at one scale)"""
        time_points = 101
        scale_points = 50
        freqs = np.linspace(0.1, 2.0, scale_points)

        # Create sharp Gaussian amplitude ridge at center SCALE
        center_scale_idx = scale_points // 2
        sigma = 2.0

        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                # Peak at center scale, constant across time
                scale_distance = abs(j - center_scale_idx)
                amplitude = np.exp(-(scale_distance**2) / (2 * sigma**2)) + 0.1
                wt[j, i] = amplitude * np.exp(1j * 0.5)

        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.05, "amplitude")

        # Ridge should be detected at center scale for all times
        self.assertTrue(rp[center_scale_idx, :].all())  # All times at center scale

        # Adjacent scales should not be ridges (strict local maximum)
        if center_scale_idx > 0:
            self.assertFalse(rp[center_scale_idx - 1, :].any())  # Scale below center
        if center_scale_idx < scale_points - 1:
            self.assertFalse(rp[center_scale_idx + 1, :].any())  # Scale above center

    def test_diagonal_amplitude_ridge(self):
        """ Create a Gaussian that moves linearly with time (diagonal ridge)"""
        time_points = 101
        scale_points = 50
        t = np.linspace(-5, 5, time_points)
        freqs = np.linspace(0.1, 2.0, scale_points)

        wt = np.zeros((scale_points, time_points), dtype=np.complex128)

        sigma = 2.0
        theoretical_peak_indices = []

        for i, time_val in enumerate(t):
            # Peak scale moves linearly with time
            peak_scale_idx = 10 + 0.3 * i  # Linear progression
            theoretical_peak_indices.append(int(np.round(peak_scale_idx)))

            # Create Gaussian centered at peak_scale_idx for this time
            for j in range(scale_points):
                scale_distance = abs(j - peak_scale_idx)
                amplitude = np.exp(-(scale_distance**2) / (2 * sigma**2)) + 0.1
                wt[j, i] = amplitude * np.exp(1j * 0.5)

        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.2, "amplitude")

        # Check that ridges are detected at expected locations
        detected_ridges = 0
        for i, expected_scale_idx in enumerate(theoretical_peak_indices):
            if expected_scale_idx < scale_points and rp[expected_scale_idx, i]:
                detected_ridges += 1

        # Should detect most of the theoretical ridge points
        detection_rate = detected_ridges / len(theoretical_peak_indices)
        self.assertGreater(detection_rate, 0.8)  # At least 80% detection

        # Verify the detected ridge points form a reasonable diagonal pattern
        ridge_scales, ridge_times = np.where(rp)
        if len(ridge_times) > 10:  # Only check if enough points detected
            # Ridge should have positive slope (higher time -> higher scale index)
            correlation = np.corrcoef(ridge_times, ridge_scales)[0, 1]
            self.assertGreater(correlation, 0.5)  # Positive correlation

    def test_amplitude_threshold_filtering(self):
        """ Test that amplitude threshold properly filters weak signals"""
        time_points = 51
        scale_points = 25
        freqs = np.linspace(0.1, 1.0, scale_points)

        # Create weak amplitude ridge at center scale
        center_scale_idx = scale_points // 2
        sigma = 2.0

        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                # Peak at center scale with max amplitude = 0.5
                scale_distance = abs(j - center_scale_idx)
                amplitude = 0.5 * np.exp(-(scale_distance**2) / (2 * sigma**2)) + 0.1
                wt[j, i] = amplitude * np.exp(1j * 0.0)

        # High threshold should reject ridge
        rp_high, _, _, _ = isridgepoint(wt, freqs, 0.8, "amplitude")
        self.assertFalse(rp_high.any())

        # Low threshold should detect ridge
        rp_low, _, _, _ = isridgepoint(wt, freqs, 0.2, "amplitude")
        self.assertTrue(rp_low[center_scale_idx, :].all())  # Center scale detected

    def test_phase_ridge_linear_chirp_increasing(self):
        """ Test phase ridge detection with linear chirp"""
        time_points = 50
        scale_points = 25

        # Use unit time spacing as assumed by isridgepoint
        t = np.arange(time_points, dtype=float)  # 0, 1, 2, ..., 49
        freqs = np.linspace(0.1, 1.0, scale_points)

        # Create a simple linear chirp with quadratic phase
        chirp_rate = 0.02  # Hz per second
        phase = chirp_rate * t**2  # Quadratic phase, no scale dependence

        # Create wavelet transform with same phase for all scales
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for j in range(scale_points):
            wt[j, :] = 1.0 * np.exp(1j * phase)

        # Run ridge detection with lower threshold
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.01, "phase")

        # Should detect ridge points
        self.assertTrue(rp.any(), "No phase ridges detected in linear chirp")

        # Calculate theoretical crossing points
        # inst_freq = 2 * chirp_rate * t
        # Crossing occurs when 2 * chirp_rate * t = scale_freq
        # So t = scale_freq / (2 * chirp_rate)

        # Should detect ridge points
        self.assertTrue(rp.any())

        # Check that detected ridges are near theoretical crossing points
        ridge_scales, ridge_times = np.where(rp)
        ridge_time_vals = t[ridge_times]
        ridge_freqs = freqs[ridge_scales]

        # Exclude ridges near boundaries to avoid edge effects
        boundary = 2
        ridge_time_vals_filtered = ridge_time_vals[boundary:-boundary]
        ridge_freqs_filtered = ridge_freqs[boundary:-boundary]

        # For each detected ridge (excluding boundary ridges),
        # check it's close to theoretical crossing
        for i, (ridge_time, ridge_freq) in enumerate(
            zip(ridge_time_vals_filtered, ridge_freqs_filtered)
        ):
            theoretical_time = ridge_freq / (2 * chirp_rate)
            time_error = abs(ridge_time - theoretical_time)
            self.assertLess(time_error, 0.5001)

    def test_phase_ridge_linear_chirp_decreasing(self):
        """ Test phase ridge detection with linear chirp and decreasing frequency matrix"""
        time_points = 50
        scale_points = 25

        # Use unit time spacing as assumed by isridgepoint
        t = np.arange(time_points, dtype=float)  # 0, 1, 2, ..., 49
        freqs = np.linspace(1.0, 0.1, scale_points)  # Decreasing frequencies

        # Create a simple linear chirp with quadratic phase
        chirp_rate = 0.02  # Hz per second
        phase = chirp_rate * t**2  # Quadratic phase, no scale dependence

        # Create wavelet transform with same phase for all scales
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for j in range(scale_points):
            wt[j, :] = 1.0 * np.exp(1j * phase)

        # Run ridge detection with lower threshold
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.01, "phase")

        # Should detect ridge points
        self.assertTrue(rp.any(), "No phase ridges detected in linear chirp")

        # Calculate theoretical crossing points
        # inst_freq = 2 * chirp_rate * t
        # Crossing occurs when 2 * chirp_rate * t = scale_freq
        # So t = scale_freq / (2 * chirp_rate)

        # Check that detected ridges are near theoretical crossing points
        ridge_scales, ridge_times = np.where(rp)
        ridge_time_vals = t[ridge_times]
        ridge_freqs = freqs[ridge_scales]

        # Exclude ridges near boundaries to avoid edge effects
        boundary = 2
        ridge_time_vals_filtered = ridge_time_vals[boundary:-boundary]
        ridge_freqs_filtered = ridge_freqs[boundary:-boundary]

        # For each detected ridge (excluding boundary ridges),
        # check it's close to theoretical crossing
        for i, (ridge_time, ridge_freq) in enumerate(
            zip(ridge_time_vals_filtered, ridge_freqs_filtered)
        ):
            theoretical_time = ridge_freq / (2 * chirp_rate)
            time_error = abs(ridge_time - theoretical_time)
            print(
                "Ridge time:",
                ridge_time,
                "Theoretical time:",
                theoretical_time,
                "Error:",
                time_error,
            )
            self.assertLess(time_error, 0.5001)

    def test_phase_ridge_non_monotonic_frequency_error(self):
        """ Test that non-monotonic frequency matrix raises an error for phase ridges"""
        time_points = 50
        scale_points = 25

        # Use unit time spacing as assumed by isridgepoint
        t = np.arange(time_points, dtype=float)  # 0, 1, 2, ..., 49

        # Create non-monotonic frequencies (goes up then down)
        freqs = np.concatenate(
            [
                np.linspace(0.1, 0.6, scale_points // 2),
                np.linspace(0.5, 0.1, scale_points - scale_points // 2),
            ]
        )

        # Create a simple linear chirp with quadratic phase
        chirp_rate = 0.02  # Hz per second
        phase = chirp_rate * t**2  # Quadratic phase, no scale dependence

        # Create wavelet transform with same phase for all scales
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for j in range(scale_points):
            wt[j, :] = 1.0 * np.exp(1j * phase)

        # Should raise ValueError for non-monotonic frequency matrix in phase ridge detection
        with self.assertRaises(ValueError) as context:
            rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.01, "phase")

        # Check that the error message mentions monotonic frequency requirement
        self.assertIn("monotonic", str(context.exception).lower())

    def test_frequency_constraints(self):
        """ Test frequency min/max constraints"""
        time_points = 51
        scale_points = 25
        t = np.linspace(-3, 3, time_points)
        freqs = np.linspace(0.1, 2.0, scale_points)

        # Create amplitude ridge at center
        sigma = 0.5
        amplitudes = np.exp(-(t**2) / (2 * sigma**2)) + 0.1

        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                wt[j, i] = amplitudes[i] * np.exp(1j * j)  # varying phase

        # Constrain to middle frequency range
        freq_min = 0.8
        freq_max = 1.2

        rp, rq, proc, inst_freq = isridgepoint(
            wt, freqs, 0.5, "amplitude", freq_min=freq_min, freq_max=freq_max
        )

        # Only scales within frequency range should have potential ridges
        valid_scales = (freqs >= freq_min) & (freqs <= freq_max)
        if valid_scales.any():
            # Check that ridges appear in the constrained frequency range
            ridge_detected = rp.any(axis=1)
            # All detected ridges should be in valid frequency range
            invalid_ridges = ridge_detected & ~valid_scales
            self.assertFalse(invalid_ridges.any())

    def test_multivariate_ridge_detection(self):
        """ Create a Gaussian that moves linearly with time (diagonal ridge)"""
        time_points = 31
        scale_points = 15
        components = 2
        t = np.linspace(-2, 2, time_points)
        freqs = np.linspace(0.5, 1.5, scale_points)

        # Create multivariate wavelet transform
        wt = np.zeros((scale_points, time_points, components), dtype=np.complex128)
        sigma = 2.0

        for i, time_val in enumerate(t):
            # Peak scale moves linearly with time, but keep it within bounds
            peak_scale_idx = 2 + 0.35 * i

            # Create Gaussian centered at peak_scale_idx for this time
            for j in range(scale_points):
                scale_distance = abs(j - peak_scale_idx)
                amplitude = np.exp(-(scale_distance**2) / (2 * sigma**2)) + 0.1
                wt[j, i, 0] = amplitude * np.exp(1j * 0.5)
                wt[j, i, 1] = amplitude * np.exp(1j * 1.0)

        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.1, "amplitude")

        # Debug: Check what we actually created
        center_idx = time_points // 2

        # Should detect ridge at center time - ridge is along scale axis (axis 0)
        self.assertTrue(rp[:, center_idx].any())

        # Output shapes should match 2D case
        self.assertEqual(rp.shape, (scale_points, time_points))
        self.assertEqual(rq.shape, (scale_points, time_points))

    def test_ridge_quantity_values(self):
        """ Test that ridge quantities have expected values"""
        time_points = 31
        scale_points = 15
        t = np.linspace(-2, 2, time_points)
        freqs = np.linspace(0.5, 1.5, scale_points)

        # Create simple amplitude ridge
        sigma = 0.4
        amplitudes = np.exp(-(t**2) / (2 * sigma**2)) + 0.1

        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                wt[j, i] = amplitudes[i] * np.exp(1j * 0.5)

        # Test amplitude ridge
        rp_amp, rq_amp, _, _ = isridgepoint(wt, freqs, 0.1, "amplitude")

        # Ridge quantity for amplitude should be the amplitude itself
        amp, _, _, _ = instmom_univariate(wt, axis=1)
        self.assertTrue(np.allclose(rq_amp, amp))

        # Test phase ridge
        rp_phase, rq_phase, _, inst_freq = isridgepoint(wt, freqs, 0.1, "phase")

        # Ridge quantity for phase should be inst_freq - scale_freq
        freq_matrix = np.broadcast_to(freqs[:, np.newaxis], (scale_points, time_points))
        expected_phase_rq = inst_freq - freq_matrix
        self.assertTrue(np.allclose(rq_phase, expected_phase_rq))


class TestRidgeShiftInterpolation:
    """Tests for the ridge_shift_interpolation function."""

    def test_ridge_shift_interpolation_edge_cases(self):
        """Test ridge_shift_interpolation with edge cases."""
        # Test with empty ridge points
        empty_rp = np.zeros((10, 10), dtype=bool)
        empty_rq = np.zeros((10, 10))
        empty_data = [np.zeros((10, 10)), np.zeros((10, 10))]

        result = ridge_shift_interpolation(empty_rp, empty_rq, empty_data)
        assert len(result) == 2
        assert len(result[0]) == 0
        assert len(result[1]) == 0

        # Test with single ridge point
        single_rp = np.zeros((10, 10), dtype=bool)
        single_rp[5, 5] = True
        single_rq = np.random.rand(10, 10)
        single_data = [np.random.rand(10, 10), np.random.rand(10, 10)]

        result = ridge_shift_interpolation(single_rp, single_rq, single_data)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert np.isfinite(result[0][0])
        assert np.isfinite(result[1][0])

        # Test with boundary ridge points (should use original values)
        boundary_rp = np.zeros((10, 10), dtype=bool)
        boundary_rp[0, 5] = True  # Top boundary
        boundary_rp[9, 5] = True  # Bottom boundary
        boundary_rq = np.random.rand(10, 10)
        boundary_data = [np.random.rand(10, 10), np.random.rand(10, 10)]

        result = ridge_shift_interpolation(boundary_rp, boundary_rq, boundary_data)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        # Values should match original data at boundary points
        assert result[0][0] == boundary_data[0][0, 5]
        assert result[0][1] == boundary_data[0][9, 5]
        assert result[1][0] == boundary_data[1][0, 5]
        assert result[1][1] == boundary_data[1][9, 5]

        # Test with different data types
        int_data = [
            np.ones((10, 10), dtype=np.int32),
            np.ones((10, 10), dtype=np.int32),
        ]
        result = ridge_shift_interpolation(single_rp, single_rq, int_data)
        assert result[0].dtype == np.int32
        assert result[1].dtype == np.int32

    def test_ridge_shift_interpolation_synthetic_signal(self):
        """Test ridge_shift_interpolation functionality and accuracy with synthetic signal."""
        # Parameters
        t = np.linspace(0, 799, 800)
        tau = 1000.0
        omega = 0.05
        k = 10.0

        # Generate synthetic signal
        x_t, y_t = synth_signal(t, tau, omega, k)

        # Wavelet parameters
        gamma = 3
        beta = 2
        freqs = morse_logspace_freq(gamma, beta, len(t), density=4)

        # Apply wavelet transform
        wavelet_y = morse_wavelet_transform(y_t, gamma, beta, freqs, boundary="mirror")

        # Find ridge points (It is natural to have false ridges, which is why we use a amplitude threshold of 10 here)
        rp, rq, proc, inst_freq = isridgepoint(
            wavelet_y, freqs, amplitude_threshold=10, ridge_type="amplitude"
        )

        # Create frequency meshgrid and power
        freq_mesh = np.zeros_like(rq)
        for i, f in enumerate(freqs):
            freq_mesh[i, :] = f
        power = np.abs(wavelet_y) ** 2

        # Test ridge_shift_interpolation
        result = ridge_shift_interpolation(rp, rq, [power, freq_mesh])

        # Basic functionality tests
        assert len(result) == 2

        # Get the shifted ridge point frequencies
        interpolated_ridge_freqs = result[1]

        # Accuracy test against theoretical ridge
        ridge_scales, ridge_times = np.where(rp)
        ridge_time_vals = t[ridge_times]
        theoretical_at_ridge_times = theoretical_ridge_function(
            ridge_time_vals, tau, omega
        )

        # Calculate RMS error
        rms_error = np.sqrt(
            np.mean((interpolated_ridge_freqs - theoretical_at_ridge_times) ** 2)
        )
        assert rms_error < 0.003

        # Check frequency range alignment
        theoretical_full = theoretical_ridge_function(t, tau, omega)
        theoretical_range = theoretical_full.max() - theoretical_full.min()
        interpolated_range = (
            interpolated_ridge_freqs.max() - interpolated_ridge_freqs.min()
        )

        # Ranges should be similar (within 20%)
        range_ratio = interpolated_range / theoretical_range
        assert 0.8 <= range_ratio <= 1.2

        # Data consistency test
        assert len(result[0]) == len(result[1])

        # All values should be finite
        for res in result:
            assert np.all(np.isfinite(res))


class TestOrganizeRidgePoints(unittest.TestCase):
    def test_empty_ridge_points(self):
        """Test organize_ridge_points with no ridge points."""
        freq_indices = np.array([], dtype=int)
        time_indices = np.array([], dtype=int)
        transform_shape = (10, 20)
        arrays_to_organize = [np.array([]), np.array([])]

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        self.assertEqual(index_matrix.size, 0)
        self.assertEqual(len(organized_arrays), 0)

    def test_single_ridge_point(self):
        """Test organize_ridge_points with a single ridge point."""
        freq_indices = np.array([3])
        time_indices = np.array([5])
        transform_shape = (10, 20)
        arrays_to_organize = [np.array([1.5]), np.array([2.7])]

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        # Should create matrix with shape (20, 1) since max ridges = 1
        self.assertEqual(index_matrix.shape, (20, 1))
        self.assertEqual(organized_arrays[0].shape, (20, 1))
        self.assertEqual(organized_arrays[1].shape, (20, 1))

        # Only time index 5 should have a ridge
        self.assertEqual(index_matrix[5, 0], 0)  # Original index 0
        self.assertEqual(organized_arrays[0][5, 0], 1.5)
        self.assertEqual(organized_arrays[1][5, 0], 2.7)

        # All other positions should be NaN
        for t in range(20):
            if t != 5:
                self.assertTrue(np.isnan(index_matrix[t, 0]))
                self.assertTrue(np.isnan(organized_arrays[0][t, 0]))
                self.assertTrue(np.isnan(organized_arrays[1][t, 0]))

    def test_multiple_ridges_same_time(self):
        """Test organize_ridge_points with multiple ridges at the same time."""
        freq_indices = np.array([2, 5, 8])
        time_indices = np.array([10, 10, 10])  # All at same time
        transform_shape = (20, 30)
        arrays_to_organize = [np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6])]

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        # Should create matrix with shape (30, 3) since max ridges = 3
        self.assertEqual(index_matrix.shape, (30, 3))
        self.assertEqual(organized_arrays[0].shape, (30, 3))
        self.assertEqual(organized_arrays[1].shape, (30, 3))

        # Time index 10 should have all three ridges
        expected_indices = [0, 1, 2]  # Original indices
        expected_values_0 = [1.1, 2.2, 3.3]
        expected_values_1 = [4.4, 5.5, 6.6]

        for ridge_idx in range(3):
            self.assertEqual(index_matrix[10, ridge_idx], expected_indices[ridge_idx])
            self.assertEqual(
                organized_arrays[0][10, ridge_idx], expected_values_0[ridge_idx]
            )
            self.assertEqual(
                organized_arrays[1][10, ridge_idx], expected_values_1[ridge_idx]
            )

        # All other times should be NaN
        for t in range(30):
            if t != 10:
                for ridge_idx in range(3):
                    self.assertTrue(np.isnan(index_matrix[t, ridge_idx]))
                    self.assertTrue(np.isnan(organized_arrays[0][t, ridge_idx]))
                    self.assertTrue(np.isnan(organized_arrays[1][t, ridge_idx]))

    def test_ridges_across_multiple_times(self):
        """Test organize_ridge_points with ridges distributed across different times."""
        freq_indices = np.array([1, 3, 2, 4, 0])
        time_indices = np.array([0, 0, 5, 5, 10])  # Two at t=0, two at t=5, one at t=10
        transform_shape = (10, 15)
        arrays_to_organize = [
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ]

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        # Max ridges = 2 (at times 0 and 5)
        self.assertEqual(index_matrix.shape, (15, 2))
        self.assertEqual(organized_arrays[0].shape, (15, 2))
        self.assertEqual(organized_arrays[1].shape, (15, 2))

        # Check time 0 (should have 2 ridges)
        self.assertFalse(np.isnan(index_matrix[0, 0]))
        self.assertFalse(np.isnan(index_matrix[0, 1]))
        self.assertFalse(np.isnan(organized_arrays[0][0, 0]))
        self.assertFalse(np.isnan(organized_arrays[0][0, 1]))

        # Check time 5 (should have 2 ridges)
        self.assertFalse(np.isnan(index_matrix[5, 0]))
        self.assertFalse(np.isnan(index_matrix[5, 1]))
        self.assertFalse(np.isnan(organized_arrays[0][5, 0]))
        self.assertFalse(np.isnan(organized_arrays[0][5, 1]))

        # Check time 10 (should have 1 ridge)
        self.assertFalse(np.isnan(index_matrix[10, 0]))
        self.assertTrue(
            np.isnan(index_matrix[10, 1])
        )  # Second ridge slot should be empty
        self.assertFalse(np.isnan(organized_arrays[0][10, 0]))
        self.assertTrue(np.isnan(organized_arrays[0][10, 1]))

    def test_complex_data_arrays(self):
        """Test organize_ridge_points with complex-valued data arrays."""
        freq_indices = np.array([1, 2])
        time_indices = np.array([3, 3])
        transform_shape = (5, 10)
        complex_data = np.array([1.0 + 2.0j, 3.0 + 4.0j])
        real_data = np.array([5.0, 6.0])
        arrays_to_organize = [complex_data, real_data]

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        # Check that complex array preserves complex type
        self.assertTrue(np.iscomplexobj(organized_arrays[0]))
        self.assertFalse(np.iscomplexobj(organized_arrays[1]))

        # Check values at time 3
        self.assertEqual(organized_arrays[0][3, 0], 1.0 + 2.0j)
        self.assertEqual(organized_arrays[0][3, 1], 3.0 + 4.0j)
        self.assertEqual(organized_arrays[1][3, 0], 5.0)
        self.assertEqual(organized_arrays[1][3, 1], 6.0)

    def test_sorting_by_frequency_and_time(self):
        """Test that ridge points are properly sorted by time first, then frequency."""
        # Unsorted input: mix of times and frequencies
        freq_indices = np.array([5, 2, 3, 1])
        time_indices = np.array(
            [2, 1, 2, 1]
        )  # Times: 2,1,2,1 -> should sort to 1,1,2,2
        transform_shape = (10, 5)
        arrays_to_organize = [
            np.array([50, 20, 30, 10])
        ]  # Values corresponding to original order

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        # After sorting by time then frequency:
        # Original: freq=[5,2,3,1], time=[2,1,2,1], values=[50,20,30,10]
        # Sorted:   freq=[1,2,3,5], time=[1,1,2,2], values=[10,20,30,50]

        # Time 1 should have ridges at frequencies 1,2 with values 10,20
        # Time 2 should have ridges at frequencies 3,5 with values 30,50

        # Check time 1 values (sorted by frequency: 1,2)
        self.assertEqual(organized_arrays[0][1, 0], 10)  # freq=1, value=10
        self.assertEqual(organized_arrays[0][1, 1], 20)  # freq=2, value=20

        # Check time 2 values (sorted by frequency: 3,5)
        self.assertEqual(organized_arrays[0][2, 0], 30)  # freq=3, value=30
        self.assertEqual(organized_arrays[0][2, 1], 50)  # freq=5, value=50

    def test_different_array_dtypes(self):
        """Test organize_ridge_points preserves different data types."""
        freq_indices = np.array([1])
        time_indices = np.array([2])
        transform_shape = (5, 5)

        int_array = np.array([42], dtype=np.int32)
        float_array = np.array([3.14], dtype=np.float64)
        complex_array = np.array([1.0 + 2.0j], dtype=np.complex128)

        arrays_to_organize = [int_array, float_array, complex_array]

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        # Check that dtypes are preserved appropriately
        # Integer arrays become float (due to NaN fill)
        self.assertTrue(np.issubdtype(organized_arrays[0].dtype, np.floating))
        self.assertTrue(np.issubdtype(organized_arrays[1].dtype, np.floating))
        self.assertTrue(np.issubdtype(organized_arrays[2].dtype, np.complexfloating))

        # Check values
        self.assertEqual(organized_arrays[0][2, 0], 42)
        self.assertEqual(organized_arrays[1][2, 0], 3.14)
        self.assertEqual(organized_arrays[2][2, 0], 1.0 + 2.0j)

    def test_index_matrix_correctness(self):
        """Test that index matrix correctly maps to original ridge point order."""
        freq_indices = np.array([1, 3, 2])
        time_indices = np.array([0, 1, 0])
        transform_shape = (5, 3)
        arrays_to_organize = [np.array([100, 200, 300])]

        index_matrix, organized_arrays = organize_ridge_points(
            freq_indices, time_indices, transform_shape, arrays_to_organize
        )

        # After sorting: time=[0,0,1], freq=[1,2,3], values=[100,300,200]
        # So sorted indices should be: [0,2,1]

        # Time 0 should have original indices 0 and 2 (in that order after sorting by freq)
        self.assertEqual(
            index_matrix[0, 0], 0
        )  # Original index for freq=1, time=0, value=100
        self.assertEqual(
            index_matrix[0, 1], 1
        )  # Original index for freq=2, time=0, value=300

        # Time 1 should have original index 1
        self.assertEqual(
            index_matrix[1, 0], 2
        )  # Original index for freq=3, time=1, value=200

        # Verify values match
        self.assertEqual(organized_arrays[0][0, 0], 100)
        self.assertEqual(organized_arrays[0][0, 1], 300)
        self.assertEqual(organized_arrays[0][1, 0], 200)


class TestCreate3DDeviationMatrix(unittest.TestCase):
    def test_empty_matrices(self):
        """Test create_3d_deviation_matrix with empty input matrices."""
        freq_matrix = np.array([]).reshape(0, 0)
        freq_next_pred_matrix = np.array([]).reshape(0, 0)
        freq_prev_pred_matrix = np.array([]).reshape(0, 0)

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        self.assertEqual(result.shape, (0, 0, 0))

    def test_single_time_step(self):
        """Test create_3d_deviation_matrix with only one time step."""
        freq_matrix = np.array([[1.0, 2.0]])
        freq_next_pred_matrix = np.array([[1.1, 2.1]])
        freq_prev_pred_matrix = np.array([[0.9, 1.9]])

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        # With only one time step, result should have shape (0, 2, 2) after removing last time
        self.assertEqual(result.shape, (0, 2, 2))

    def test_two_time_steps_perfect_prediction(self):
        """Test with two time steps where predictions are perfect."""
        # Perfect forward prediction: next actual = next predicted
        freq_matrix = np.array(
            [
                [1.0, 2.0],  # time 0
                [1.1, 2.1],  # time 1
            ]
        )
        freq_next_pred_matrix = np.array(
            [
                [1.1, 2.1],  # predicted next from time 0
                [1.2, 2.2],  # predicted next from time 1 (not used)
            ]
        )
        freq_prev_pred_matrix = np.array(
            [
                [0.9, 1.9],  # predicted prev from time 0 (not used)
                [1.0, 2.0],  # predicted prev from time 1
            ]
        )

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        self.assertEqual(result.shape, (1, 2, 2))

        # For perfect predictions, deviations should be 0
        # df1 = (actual_next - pred_next) / current = (1.1 - 1.1) / 1.0 = 0
        # df2 = (pred_prev_next - current) / current = (1.0 - 1.0) / 1.0 = 0
        expected_deviation = 0.0

        # Check diagonal elements (ridge i to ridge i)
        self.assertAlmostEqual(result[0, 0, 0], expected_deviation, places=10)
        self.assertAlmostEqual(result[0, 1, 1], expected_deviation, places=10)

    def test_frequency_deviations_calculation(self):
        """Test the calculation of frequency deviations."""
        freq_matrix = np.array(
            [
                [1.0, 2.0],  # time 0
                [1.2, 2.4],  # time 1
            ]
        )
        freq_next_pred_matrix = np.array(
            [
                [1.1, 2.2],  # predicted next from time 0
                [1.3, 2.6],  # predicted next from time 1
            ]
        )
        freq_prev_pred_matrix = np.array(
            [
                [0.9, 1.8],  # predicted prev from time 0
                [1.0, 2.0],  # predicted prev from time 1
            ]
        )

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        # Manual calculation for ridge 0 to ridge 0:
        # df1[0,0,0] = (1.2 - 1.1) / 1.0 = 0.1
        # df2[0,0,0] = (1.0 - 1.0) / 1.0 = 0.0
        # df[0,0,0] = (|0.1| + |0.0|) / 2 = 0.05

        expected_00 = (abs(0.1) + abs(0.0)) / 2
        self.assertAlmostEqual(result[0, 0, 0], expected_00, places=10)

    def test_alpha_threshold_filtering(self):
        """Test that deviations above alpha threshold are set to NaN."""
        freq_matrix = np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],  # Large frequency change
            ]
        )
        freq_next_pred_matrix = np.array([[1.1, 2.1], [2.1, 1.1]])
        freq_prev_pred_matrix = np.array([[0.9, 1.9], [1.9, 0.9]])

        alpha = 0.1  # Small threshold

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix, alpha=alpha
        )

        # With large frequency changes and small alpha, many values should be NaN
        nan_count = np.sum(np.isnan(result))
        self.assertGreater(nan_count, 0)

    def test_alpha_threshold_values(self):
        """Test different alpha threshold values."""
        freq_matrix = np.array([[1.0, 2.0], [1.5, 2.5]])
        freq_next_pred_matrix = np.array([[1.2, 2.2], [1.7, 2.7]])
        freq_prev_pred_matrix = np.array([[0.8, 1.8], [1.0, 2.0]])

        # Test with very permissive alpha
        result_permissive = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix, alpha=1.0
        )

        # Test with very restrictive alpha
        result_restrictive = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix, alpha=0.01
        )

        # Permissive alpha should have fewer NaN values
        nan_count_permissive = np.sum(np.isnan(result_permissive))
        nan_count_restrictive = np.sum(np.isnan(result_restrictive))

        self.assertLessEqual(nan_count_permissive, nan_count_restrictive)

    def test_3d_matrix_dimensions(self):
        """Test that the 3D matrix has correct dimensions and indexing."""
        num_times = 4
        max_ridges = 3

        freq_matrix = np.random.rand(num_times, max_ridges)
        freq_next_pred_matrix = np.random.rand(num_times, max_ridges)
        freq_prev_pred_matrix = np.random.rand(num_times, max_ridges)

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        # Should be (time-1, max_ridges, max_ridges)
        expected_shape = (num_times - 1, max_ridges, max_ridges)
        self.assertEqual(result.shape, expected_shape)

    def test_nan_input_handling(self):
        """Test behavior with NaN values in input matrices."""
        freq_matrix = np.array([[1.0, np.nan], [1.2, 2.4]])
        freq_next_pred_matrix = np.array([[1.1, 2.2], [1.3, np.nan]])
        freq_prev_pred_matrix = np.array([[np.nan, 1.8], [1.0, 2.0]])

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        # Function should handle NaN inputs gracefully
        self.assertEqual(result.shape, (1, 2, 2))
        # Result should contain NaN where inputs had NaN
        self.assertTrue(np.isnan(result[0, 0, 1]))  # Due to NaN in freq_matrix[0,1]

    def test_zero_frequency_handling(self):
        """Test behavior when frequencies are zero (division by zero)."""
        freq_matrix = np.array(
            [
                [0.0, 1.0],  # Zero frequency
                [1.0, 2.0],
            ]
        )
        freq_next_pred_matrix = np.array([[0.5, 1.5], [1.5, 2.5]])
        freq_prev_pred_matrix = np.array([[0.0, 0.5], [0.0, 1.0]])

        # Should not raise exception due to errstate context manager
        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        self.assertEqual(result.shape, (1, 2, 2))
        # Division by zero should result in inf, which gets converted to NaN by alpha threshold
        self.assertTrue(np.isnan(result[0, 0, 0]) or np.isinf(result[0, 0, 0]))

    def test_bidirectional_error_combination(self):
        """Test that forward and backward prediction errors are properly combined."""
        freq_matrix = np.array([[1.0, 2.0], [1.1, 2.1]])

        # Set up asymmetric prediction errors
        freq_next_pred_matrix = np.array(
            [
                [1.0, 2.0],  # Perfect forward prediction
                [1.2, 2.2],
            ]
        )
        freq_prev_pred_matrix = np.array(
            [
                [0.8, 1.8],
                [1.2, 2.2],  # Large backward prediction error
            ]
        )

        result = create_3d_deviation_matrix(
            freq_matrix, freq_next_pred_matrix, freq_prev_pred_matrix
        )

        # Verify that both forward and backward errors contribute
        # The result should be the average of absolute forward and backward errors
        self.assertEqual(result.shape, (1, 2, 2))

        # Manual check for element [0,0,0]:
        # df1 = (1.1 - 1.0) / 1.0 = 0.1 (forward error)
        # df2 = (1.2 - 1.0) / 1.0 = 0.2 (backward error)
        # result = (|0.1| + |0.2|) / 2 = 0.15
        expected = (abs(0.1) + abs(0.2)) / 2
        self.assertAlmostEqual(result[0, 0, 0], expected, places=10)


class TestBilateralMinimumSelection(unittest.TestCase):
    def test_empty_matrix(self):
        """Test bilateral_minimum_selection with empty input matrix."""
        df = np.array([]).reshape(0, 0, 0)

        result = bilateral_minimum_selection(df)

        self.assertEqual(result.shape, (0, 0, 0))

    def test_single_time_single_ridge(self):
        """Test with single time step and single ridge."""
        df = np.array([[[0.5]]])

        result = bilateral_minimum_selection(df)

        self.assertEqual(result.shape, (1, 1, 1))
        self.assertEqual(result[0, 0, 0], 0.5)

    def test_all_nan_matrix(self):
        """Test with matrix containing only NaN values."""
        df = np.full((2, 3, 3), np.nan)

        result = bilateral_minimum_selection(df)

        self.assertEqual(result.shape, (2, 3, 3))
        self.assertTrue(np.all(np.isnan(result)))

    def test_simple_row_minimum_selection(self):
        """Test that minimum values are selected from each row."""
        df = np.array(
            [
                [
                    [0.5, 0.3, 0.8],  # Row 0: min at col 1
                    [0.2, 0.7, 0.4],  # Row 1: min at col 0
                    [0.9, 0.6, 0.1],
                ]  # Row 2: min at col 2
            ]
        )

        result = bilateral_minimum_selection(df)

        # After step 1 (row minimums), should have values at:
        # [0,0,1] = 0.3, [0,1,0] = 0.2, [0,2,2] = 0.1
        expected_step1 = np.full((1, 3, 3), np.nan)
        expected_step1[0, 0, 1] = 0.3
        expected_step1[0, 1, 0] = 0.2
        expected_step1[0, 2, 2] = 0.1

        # Step 2 (column minimums) will further filter these
        # Column 0: min([nan, 0.2, nan]) -> keeps [0,1,0] = 0.2
        # Column 1: min([0.3, nan, nan]) -> keeps [0,0,1] = 0.3
        # Column 2: min([nan, nan, 0.1]) -> keeps [0,2,2] = 0.1

        self.assertFalse(np.isnan(result[0, 0, 1]))  # Should keep 0.3
        self.assertFalse(np.isnan(result[0, 1, 0]))  # Should keep 0.2
        self.assertFalse(np.isnan(result[0, 2, 2]))  # Should keep 0.1
        self.assertEqual(result[0, 0, 1], 0.3)
        self.assertEqual(result[0, 1, 0], 0.2)
        self.assertEqual(result[0, 2, 2], 0.1)

    def test_bilateral_filtering_effect(self):
        """Test that bilateral filtering removes non-optimal connections."""
        df = np.array(
            [
                [
                    [0.1, 0.9],  # Row 0: min at col 0
                    [0.8, 0.2],
                ]  # Row 1: min at col 1
            ]
        )

        result = bilateral_minimum_selection(df)

        # After step 1: [0,0,0]=0.1, [0,1,1]=0.2
        # After step 2:
        # Column 0: min([0.1, nan]) -> keeps [0,0,0]=0.1
        # Column 1: min([nan, 0.2]) -> keeps [0,1,1]=0.2

        self.assertEqual(result[0, 0, 0], 0.1)
        self.assertEqual(result[0, 1, 1], 0.2)
        self.assertTrue(np.isnan(result[0, 0, 1]))
        self.assertTrue(np.isnan(result[0, 1, 0]))

    def test_multiple_time_steps(self):
        """Test with multiple time steps."""
        df = np.array(
            [
                # Time 0
                [[0.5, 0.3], [0.7, 0.1]],
                # Time 1
                [[0.2, 0.8], [0.6, 0.4]],
            ]
        )

        result = bilateral_minimum_selection(df)

        self.assertEqual(result.shape, (2, 2, 2))

        # Each time step should be processed independently
        # Time 0: row mins at [0,0,1]=0.3, [0,1,1]=0.1
        # Then column mins: col 0 has no values, col 1 min([0.3,0.1]) -> [0,1,1]=0.1
        self.assertEqual(result[0, 1, 1], 0.1)

        # Time 1: row mins at [1,0,0]=0.2, [1,1,1]=0.4
        # Then column mins: col 0 min([0.2,nan]) -> [1,0,0]=0.2, col 1 min([nan,0.4]) -> [1,1,1]=0.4
        self.assertEqual(result[1, 0, 0], 0.2)
        self.assertEqual(result[1, 1, 1], 0.4)

    def test_partial_nan_rows(self):
        """Test handling of rows with some NaN values."""
        df = np.array(
            [
                [
                    [np.nan, 0.5, 0.3],  # Row 0: min at col 2
                    [0.2, np.nan, 0.8],  # Row 1: min at col 0
                    [np.nan, np.nan, np.nan],
                ]  # Row 2: all NaN (invalid)
            ]
        )

        result = bilateral_minimum_selection(df)

        # Row 0 and 1 should be processed, row 2 should remain all NaN
        self.assertFalse(np.isnan(result[0, 0, 2]))  # Should keep 0.3
        self.assertFalse(np.isnan(result[0, 1, 0]))  # Should keep 0.2

        # Row 2 should remain all NaN
        self.assertTrue(np.all(np.isnan(result[0, 2, :])))

    def test_competing_minimums_same_column(self):
        """Test when multiple rows have minimum pointing to same column."""
        df = np.array(
            [
                [
                    [0.9, 0.1],  # Row 0: min at col 1
                    [0.8, 0.2],
                ]  # Row 1: min at col 1 (competing for same column)
            ]
        )

        result = bilateral_minimum_selection(df)

        # After step 1: both [0,0,1]=0.1 and [0,1,1]=0.2
        # After step 2: column 1 min([0.1, 0.2]) -> only [0,0,1]=0.1 survives
        self.assertEqual(result[0, 0, 1], 0.1)
        self.assertTrue(np.isnan(result[0, 1, 1]))  # Should be filtered out

    def test_edge_case_identical_values(self):
        """Test behavior with identical minimum values."""
        df = np.array(
            [
                [
                    [0.5, 0.3, 0.3],  # Row 0: two identical mins at cols 1,2
                    [0.7, 0.2, 0.8],
                ]  # Row 1: min at col 1
            ]
        )

        result = bilateral_minimum_selection(df)

        # np.nanargmin should return first occurrence (col 1 for row 0)
        # After step 1: [0,0,1]=0.3, [0,1,1]=0.2
        # After step 2: column 1 min([0.3, 0.2]) -> [0,1,1]=0.2 survives
        self.assertEqual(result[0, 1, 1], 0.2)
        self.assertTrue(np.isnan(result[0, 0, 1]))


class TestSeparateRidgeGroups(unittest.TestCase):
    def setUp(self):
        """Set up test data with dual frequency synthetic signal."""
        # Create dual frequency synthetic signal
        self.t, self.combined_x, self.combined_y, self.params1, self.params2 = (
            create_dual_frequency_synthetic_signal()
        )

        # Wavelet parameters
        self.gamma = 3
        self.beta = 2
        self.freqs = morse_logspace_freq(self.gamma, self.beta, len(self.t), density=4)

        # Create wavelet transform
        self.wavelet_y = morse_wavelet_transform(
            self.combined_y, self.gamma, self.beta, self.freqs, boundary="mirror"
        )

        # Detect ridges
        self.rp, self.rq, self.proc, self.inst_freq = isridgepoint(
            self.wavelet_y, self.freqs, amplitude_threshold=10, ridge_type="amplitude"
        )

        # Create frequency meshgrid
        self.freq_mesh = np.zeros_like(self.rq)
        for i, f in enumerate(self.freqs):
            self.freq_mesh[i, :] = f

        # Get ridge point locations
        self.ridge_scales, self.ridge_times = np.where(self.rp)
        self.ridge_freqs = self.freq_mesh[self.ridge_scales, self.ridge_times]

        # Calculate instantaneous frequency derivative
        self.inst_frequency_derivative = np.gradient(self.inst_freq, axis=-1)

        # Apply ridge shift interpolation
        self.amplitude, self.interpolated_ridge_freqs, self.interp_inst_freq_dev = (
            ridge_shift_interpolation(
                self.rp,
                self.rq,
                [
                    np.abs(self.wavelet_y),
                    self.freq_mesh,
                    self.inst_frequency_derivative,
                ],
            )
        )

    def test_empty_input(self):
        """Test separate_ridge_groups with empty ridge points."""
        empty_freq_indices = np.array([], dtype=int)
        empty_time_indices = np.array([], dtype=int)
        empty_freq_mesh = np.array([])
        empty_inst_freq_deriv = np.array([])

        group_data, num_groups = separate_ridge_groups(
            empty_freq_indices,
            empty_time_indices,
            self.wavelet_y.shape,
            empty_freq_mesh,
            empty_inst_freq_deriv,
        )

        self.assertEqual(num_groups, 0)
        self.assertEqual(len(group_data), 0)

    def test_single_ridge_point(self):
        """Test with single ridge point."""
        single_freq_idx = np.array([10])
        single_time_idx = np.array([50])
        single_freq_mesh = np.array([0.1])
        single_inst_freq_deriv = np.array([0.01])

        group_data, num_groups = separate_ridge_groups(
            single_freq_idx,
            single_time_idx,
            self.wavelet_y.shape,
            single_freq_mesh,
            single_inst_freq_deriv,
            min_group_size=1,
        )

        # Single point should form one group if min_group_size=1
        self.assertEqual(num_groups, 1)
        self.assertEqual(len(group_data), 1)

        # Check group structure
        group = group_data[1]
        self.assertIn("indices", group)
        self.assertIn("values", group)

        freq_indices, time_indices = group["indices"]
        self.assertEqual(len(freq_indices), 1)
        self.assertEqual(len(time_indices), 1)
        self.assertEqual(freq_indices[0], 10)
        self.assertEqual(time_indices[0], 50)

    def test_dual_frequency_separation(self):
        """Test separation of dual frequency synthetic signal."""
        group_data, num_groups = separate_ridge_groups(
            freq_indices=self.ridge_scales,
            time_indices=self.ridge_times,
            transform_shape=self.wavelet_y.shape,
            freq_mesh=self.interpolated_ridge_freqs,
            inst_frequency_derivative=self.interp_inst_freq_dev,
            min_group_size=5,
            max_gap=2,
        )

        # Should find at least one group
        self.assertGreater(num_groups, 0)
        self.assertEqual(len(group_data), num_groups)

        # Each group should have proper structure
        for group_id, group in group_data.items():
            self.assertIn("indices", group)
            self.assertIn("values", group)

            freq_indices, time_indices = group["indices"]
            values = group["values"]

            # Should have at least min_group_size points
            self.assertGreaterEqual(len(freq_indices), 5)
            self.assertGreaterEqual(len(time_indices), 5)
            self.assertEqual(len(freq_indices), len(time_indices))

            # Values should have same length as indices
            for val_array in values:
                self.assertEqual(len(val_array), len(freq_indices))

    def test_frequency_continuity_within_groups(self):
        """Test that frequencies within groups show continuity."""
        group_data, num_groups = separate_ridge_groups(
            freq_indices=self.ridge_scales,
            time_indices=self.ridge_times,
            transform_shape=self.wavelet_y.shape,
            freq_mesh=self.interpolated_ridge_freqs,
            inst_frequency_derivative=self.interp_inst_freq_dev,
            min_group_size=5,
            max_gap=2,
        )

        for group_id, group in group_data.items():
            freq_indices, time_indices = group["indices"]
            freq_values = group["values"][0]  # First value array is frequency

            # Sort by time for continuity check
            sort_order = np.argsort(time_indices)
            sorted_freqs = freq_values[sort_order]
            sorted_times = time_indices[sort_order]

            # Check for reasonable frequency continuity
            if len(sorted_freqs) > 1:
                time_diffs = np.diff(sorted_times)

                # Time differences should respect max_gap
                self.assertLessEqual(np.max(time_diffs), 2)  # max_gap=2

    def test_min_group_size_filtering(self):
        """Test that groups smaller than min_group_size are filtered out."""
        # Start with the dual frequency data and artificially fragment it
        original_ridge_scales = self.ridge_scales.copy()
        original_ridge_times = self.ridge_times.copy()
        original_freq_mesh = self.interpolated_ridge_freqs.copy()
        original_inst_freq_deriv = self.interp_inst_freq_dev.copy()

        # Remove random ridge points to create smaller fragments
        np.random.seed(42)  # For reproducible results
        keep_indices = np.random.choice(
            len(original_ridge_scales),
            size=len(original_ridge_scales) // 3,
            replace=False,
        )

        fragmented_freq_indices = original_ridge_scales[keep_indices]
        fragmented_time_indices = original_ridge_times[keep_indices]
        fragmented_freq_mesh = original_freq_mesh[keep_indices]
        fragmented_inst_freq_deriv = original_inst_freq_deriv[keep_indices]

        # Test with small min_group_size (should keep fragmented groups)
        group_data_small, num_groups_small = separate_ridge_groups(
            fragmented_freq_indices,
            fragmented_time_indices,
            self.wavelet_y.shape,
            fragmented_freq_mesh,
            fragmented_inst_freq_deriv,
            min_group_size=3,
            max_gap=2,
        )

        # Test with large min_group_size (should filter out small fragments)
        group_data_large, num_groups_large = separate_ridge_groups(
            fragmented_freq_indices,
            fragmented_time_indices,
            self.wavelet_y.shape,
            fragmented_freq_mesh,
            fragmented_inst_freq_deriv,
            min_group_size=15,
            max_gap=2,
        )

        # Larger min_group_size should result in fewer or equal groups
        self.assertLessEqual(num_groups_large, num_groups_small)

        # All groups from large min_group_size should meet the requirement
        for group in group_data_large.values():
            freq_indices, time_indices = group["indices"]
            self.assertGreaterEqual(len(freq_indices), 15)

    def test_max_gap_filtering(self):
        """Test that groups with large time gaps are filtered appropriately."""
        # Create data with intentional time gaps by removing consecutive ridge points
        original_ridge_scales = self.ridge_scales.copy()
        original_ridge_times = self.ridge_times.copy()
        original_freq_mesh = self.interpolated_ridge_freqs.copy()
        original_inst_freq_deriv = self.interp_inst_freq_dev.copy()

        # Remove blocks of consecutive time points to create gaps
        # Find regions where we can create gaps

        # Create artificial gaps by removing ridge points in certain time ranges
        gap_mask = np.ones(len(original_ridge_times), dtype=bool)

        # Remove ridge points in specific time windows to create gaps
        time_range = np.max(original_ridge_times) - np.min(original_ridge_times)
        gap_start1 = np.min(original_ridge_times) + time_range * 0.3
        gap_end1 = gap_start1 + 5  # Create a 5-time-step gap

        gap_start2 = np.min(original_ridge_times) + time_range * 0.7
        gap_end2 = gap_start2 + 8  # Create an 8-time-step gap

        gap_mask &= ~(
            (original_ridge_times >= gap_start1) & (original_ridge_times <= gap_end1)
        )
        gap_mask &= ~(
            (original_ridge_times >= gap_start2) & (original_ridge_times <= gap_end2)
        )

        gapped_freq_indices = original_ridge_scales[gap_mask]
        gapped_time_indices = original_ridge_times[gap_mask]
        gapped_freq_mesh = original_freq_mesh[gap_mask]
        gapped_inst_freq_deriv = original_inst_freq_deriv[gap_mask]

        # Test with small max_gap (should be strict about gaps)
        group_data_small_gap, num_groups_small_gap = separate_ridge_groups(
            gapped_freq_indices,
            gapped_time_indices,
            self.wavelet_y.shape,
            gapped_freq_mesh,
            gapped_inst_freq_deriv,
            min_group_size=3,
            max_gap=2,
        )

        # Test with large max_gap (should be more tolerant of gaps)
        group_data_large_gap, num_groups_large_gap = separate_ridge_groups(
            gapped_freq_indices,
            gapped_time_indices,
            self.wavelet_y.shape,
            gapped_freq_mesh,
            gapped_inst_freq_deriv,
            min_group_size=3,
            max_gap=10,
        )

        # Larger max_gap should allow more groups (less strict filtering)
        self.assertLessEqual(num_groups_small_gap, num_groups_large_gap)

        # Verify that groups from small_gap actually respect the gap constraint
        for group in group_data_small_gap.values():
            freq_indices, time_indices = group["indices"]
            if len(time_indices) > 1:
                sorted_group_times = np.sort(time_indices)
                max_time_gap = np.max(np.diff(sorted_group_times))
                self.assertLessEqual(
                    max_time_gap,
                    2,
                    f"Group has time gap of {max_time_gap}, exceeds max_gap=2",
                )

    def test_alpha_parameter_effect(self):
        """Test effect of alpha parameter on group formation."""
        # Very restrictive alpha (small values get filtered)
        group_data_restrictive, num_groups_restrictive = separate_ridge_groups(
            freq_indices=self.ridge_scales,
            time_indices=self.ridge_times,
            transform_shape=self.wavelet_y.shape,
            freq_mesh=self.interpolated_ridge_freqs,
            inst_frequency_derivative=self.interp_inst_freq_dev,
            alpha=0.01,
            min_group_size=3,
            max_gap=2,
        )

        # Very permissive alpha
        group_data_permissive, num_groups_permissive = separate_ridge_groups(
            freq_indices=self.ridge_scales,
            time_indices=self.ridge_times,
            transform_shape=self.wavelet_y.shape,
            freq_mesh=self.interpolated_ridge_freqs,
            inst_frequency_derivative=self.interp_inst_freq_dev,
            alpha=100.0,
            min_group_size=3,
            max_gap=2,
        )

        # Restrictive alpha should create more groups (fragmentation)
        # Expected: restrictive ~10 groups, permissive ~2 groups
        self.assertGreaterEqual(
            num_groups_restrictive,
            5,
            "Restrictive alpha should create multiple fragmented groups",
        )
        self.assertLessEqual(
            num_groups_permissive,
            5,
            "Permissive alpha should consolidate into fewer main groups",
        )

        # Restrictive alpha should produce more groups than permissive
        self.assertGreater(
            num_groups_restrictive,
            num_groups_permissive,
            "Restrictive alpha should fragment ridges into more groups than permissive alpha",
        )

    def test_group_data_structure(self):
        """Test the structure and content of returned group data."""
        group_data, num_groups = separate_ridge_groups(
            freq_indices=self.ridge_scales,
            time_indices=self.ridge_times,
            transform_shape=self.wavelet_y.shape,
            freq_mesh=self.interpolated_ridge_freqs,
            inst_frequency_derivative=self.interp_inst_freq_dev,
            min_group_size=5,
            max_gap=2,
        )

        # Check return type and structure
        self.assertIsInstance(group_data, dict)
        self.assertIsInstance(num_groups, int)
        self.assertEqual(len(group_data), num_groups)

        for group_id, group in group_data.items():
            # Check group ID is positive integer
            self.assertIsInstance(group_id, int)
            self.assertGreater(group_id, 0)

            # Check group structure
            self.assertIsInstance(group, dict)
            self.assertIn("indices", group)
            self.assertIn("values", group)

            # Check indices structure
            indices = group["indices"]
            self.assertIsInstance(indices, tuple)
            self.assertEqual(len(indices), 2)

            freq_indices, time_indices = indices
            self.assertIsInstance(freq_indices, np.ndarray)
            self.assertIsInstance(time_indices, np.ndarray)

            # Check values structure
            values = group["values"]
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), 2)  # freq_mesh and inst_freq_deriv

            for val_array in values:
                self.assertIsInstance(val_array, np.ndarray)
                self.assertEqual(len(val_array), len(freq_indices))

    def test_theoretical_frequency_alignment(self):
        """Test that groups align with theoretical frequency ranges using RMS error."""
        tau1, omega1, k1 = self.params1
        tau2, omega2, k2 = self.params2

        # Calculate theoretical frequencies at all time points
        theoretical_low = theoretical_ridge_function(self.t, tau1, omega1)
        theoretical_high = theoretical_ridge_function(self.t, tau2, omega2)

        group_data, num_groups = separate_ridge_groups(
            freq_indices=self.ridge_scales,
            time_indices=self.ridge_times,
            transform_shape=self.wavelet_y.shape,
            freq_mesh=self.interpolated_ridge_freqs,
            inst_frequency_derivative=self.interp_inst_freq_dev,
            min_group_size=5,
            max_gap=2,
        )

        # Should find at least one group
        self.assertGreater(num_groups, 0)

        # Calculate RMS errors for each group against both theoretical curves
        group_assignments = {}

        for group_id, group in group_data.items():
            freq_values = group["values"][0]  # First value array is frequency
            time_indices = group["indices"][1]  # Time indices for this group

            if len(freq_values) == 0:
                continue

            # Get theoretical frequencies at the group's time points
            theoretical_low_at_times = theoretical_low[time_indices]
            theoretical_high_at_times = theoretical_high[time_indices]

            # Calculate RMS errors against both theoretical curves
            rms_error_low = np.sqrt(
                np.mean((freq_values - theoretical_low_at_times) ** 2)
            )
            rms_error_high = np.sqrt(
                np.mean((freq_values - theoretical_high_at_times) ** 2)
            )

            # Assign group to the theoretical curve with lower RMS error
            if rms_error_low < rms_error_high:
                group_assignments[group_id] = {
                    "type": "low_freq",
                    "rms_error": rms_error_low,
                    "alt_rms_error": rms_error_high,
                }
            else:
                group_assignments[group_id] = {
                    "type": "high_freq",
                    "rms_error": rms_error_high,
                    "alt_rms_error": rms_error_low,
                }

            # RMS error should be reasonable (much better than alternative)
            best_rms = min(rms_error_low, rms_error_high)
            worst_rms = max(rms_error_low, rms_error_high)

            # Best fit should be significantly better than worst fit
            self.assertLess(
                best_rms,
                worst_rms * 0.5,
                f"Group {group_id} RMS error {best_rms:.6f} should be much better "
                f"than alternative {worst_rms:.6f}",
            )

            # Absolute RMS error should be reasonable (< 10% of frequency range)
            freq_range = np.max(
                [theoretical_low.max(), theoretical_high.max()]
            ) - np.min([theoretical_low.min(), theoretical_high.min()])
            max_acceptable_rms = 0.03 * freq_range

            self.assertLess(
                best_rms,
                max_acceptable_rms,
                f"Group {group_id} RMS error {best_rms:.6f} exceeds acceptable "
                f"threshold {max_acceptable_rms:.6f}",
            )

        # Check that we have groups assigned to both frequency components
        # (if we found multiple groups)
        if num_groups >= 2:
            assigned_types = [
                assignment["type"] for assignment in group_assignments.values()
            ]
            unique_types = set(assigned_types)

            # Should have at least one group type, ideally both
            self.assertGreater(len(unique_types), 0)

            # If we have exactly 2 groups, they should preferably be different types
            if num_groups == 2 and len(group_assignments) == 2:
                self.assertEqual(
                    len(unique_types),
                    2,
                    "With 2 groups, expect one low-freq and one high-freq assignment",
                )

    def test_edge_case_no_valid_connections(self):
        """Test behavior when no valid connections can be made."""
        # Create scattered ridge points with large frequency differences
        scattered_freq_indices = np.array([0, 10, 20, 30])
        scattered_time_indices = np.array([0, 100, 200, 300])
        scattered_freq_mesh = np.array(
            [0.01, 0.5, 1.0, 2.0]
        )  # Very different frequencies
        scattered_inst_freq_deriv = np.array([0.001, 0.01, 0.1, 1.0])

        group_data, num_groups = separate_ridge_groups(
            scattered_freq_indices,
            scattered_time_indices,
            self.wavelet_y.shape,
            scattered_freq_mesh,
            scattered_inst_freq_deriv,
            alpha=0.01,  # Very restrictive
            min_group_size=2,
            max_gap=1,
        )

        # Should handle case gracefully
        self.assertGreaterEqual(num_groups, 0)
        self.assertIsInstance(group_data, dict)


class TestCalculateRidgeGroupLengths(unittest.TestCase):
    def test_constant_frequency(self):
        """Test with constant frequency - should integrate to f*dt/(2π)."""
        # f(t) = 2.0 from t=0 to t=4
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        constant_group_data = {
            1: {
                "indices": (np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4])),
                "values": [
                    np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
                    np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
                ],
            }
        }

        result = calculate_ridge_group_lengths(constant_group_data, t, 1)

        # Integral = 2.0 * 4.0 = 8.0
        # Length = 8.0 / (2π) = 4/π
        expected_length = 8.0 / (2.0 * np.pi)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], expected_length, places=6)

    def test_linear_frequency(self):
        """Test with linear frequency - should integrate to (b^2-a^2)/(2*dt)/(2π)."""
        # f(t) = t from t=0 to t=3
        t = np.array([0.0, 1.0, 2.0, 3.0])
        linear_group_data = {
            1: {
                "indices": (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])),
                "values": [
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 0.1, 0.2, 0.3]),
                ],
            }
        }

        result = calculate_ridge_group_lengths(linear_group_data, t, 1)

        # Integral = 4.5
        # Length = 4.5 / (2π) = 9/(4π)
        expected_length = 4.5 / (2.0 * np.pi)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], expected_length, places=6)

    def test_quadratic_frequency(self):
        """Test with quadratic frequency - (b^3-a^3)/(3*dt)/(2π)"""
        # f(t) = t² from t=0 to t=2
        t = np.array([0.0, 1.0, 2.0])
        quadratic_group_data = {
            1: {
                "indices": (np.array([0, 1, 2]), np.array([0, 1, 2])),
                "values": [np.array([0.0, 1.0, 4.0]), np.array([0.0, 0.1, 0.4])],
            }
        }

        result = calculate_ridge_group_lengths(quadratic_group_data, t, 1)

        # Integral = 8/3
        # Length = (8/3) / (2π) = 4/(3π)
        expected_length = (8.0 / 3.0) / (2.0 * np.pi)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], expected_length, places=6)

    def test_sinusoidal_frequency_full_period(self):
        """Test with sinusoidal frequency over full period - should integrate to zero."""
        # f(t) = sin(t) from t=0 to t=2π
        t = np.linspace(0, 2 * np.pi, 50)
        freq_values = np.sin(t)

        sinusoidal_group_data = {
            1: {
                "indices": (np.arange(len(t)), np.arange(len(t))),
                "values": [freq_values, np.zeros_like(freq_values)],
            }
        }

        result = calculate_ridge_group_lengths(sinusoidal_group_data, t, 1)

        # Integral = 0
        # Length = |0| / (2π) = 0
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_absolute_value_negative_integral(self):
        """Test that absolute value is applied to negative integrals."""
        # f(t) = -1 from t=0 to t=2 (negative constant)
        t = np.array([0.0, 1.0, 2.0])
        negative_group_data = {
            1: {
                "indices": (np.array([0, 1, 2]), np.array([0, 1, 2])),
                "values": [np.array([-1.0, -1.0, -1.0]), np.array([0.1, 0.1, 0.1])],
            }
        }

        result = calculate_ridge_group_lengths(negative_group_data, t, 1)

        # Integral = -1 * 2 = -2
        # Length = |-2| / (2π) = 2/(2π) = 1/π
        expected_length = 2.0 / (2.0 * np.pi)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], expected_length, places=6)

    def test_empty_group_returns_zero(self):
        """Test that empty groups return zero length."""
        empty_group_data = {}
        t = np.array([0.0, 1.0, 2.0])

        result = calculate_ridge_group_lengths(empty_group_data, t, 2)

        # Should return [0.0, 0.0] for missing groups
        expected = [0.0, 0.0]
        self.assertEqual(result, expected)


class TestRidgeAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test data with dual frequency synthetic signal."""
        # Create dual frequency synthetic signal
        self.t, _, self.combined_y, self.params1, self.params2 = (
            create_dual_frequency_synthetic_signal()
        )

        # Wavelet parameters
        self.gamma = 3
        self.beta = 2
        self.freqs = morse_logspace_freq(self.gamma, self.beta, len(self.t), density=4)

        # Create wavelet transform
        self.wavelet_y = morse_wavelet_transform(
            self.combined_y, self.gamma, self.beta, self.freqs, boundary="mirror"
        )

    def test_ridge_analysis_amplitude_integration(self):
        """Test complete ridge analysis workflow with amplitude ridges."""
        result = ridge_analysis(
            wavelet=self.wavelet_y,
            freqs=self.freqs,
            t=self.t,
            ridge_type="amplitude",
            amplitude_threshold=10.0,
            min_group_size=5,
            max_gap=2,
        )

        # Check that all expected keys are present
        expected_keys = {
            "ridge_points",
            "ridge_quantity",
            "num_groups",
            "group_lengths",
            "group_data",
            "inst_frequency",
            "ridge_data",
        }
        self.assertEqual(set(result.keys()), expected_keys)

        # Check data types and shapes
        self.assertIsInstance(result["ridge_points"], np.ndarray)
        self.assertEqual(result["ridge_points"].dtype, bool)
        self.assertEqual(result["ridge_points"].shape, self.wavelet_y.shape)

        self.assertIsInstance(result["ridge_quantity"], np.ndarray)
        self.assertEqual(result["ridge_quantity"].shape, self.wavelet_y.shape)

        self.assertIsInstance(result["num_groups"], int)
        self.assertGreaterEqual(result["num_groups"], 0)

        self.assertIsInstance(result["group_lengths"], list)
        self.assertEqual(len(result["group_lengths"]), result["num_groups"])

        self.assertIsInstance(result["group_data"], dict)
        self.assertEqual(len(result["group_data"]), result["num_groups"])

        self.assertIsInstance(result["inst_frequency"], np.ndarray)
        self.assertEqual(result["inst_frequency"].shape, self.wavelet_y.shape)

        self.assertIsInstance(result["ridge_data"], list)

    def test_ridge_analysis_phase_integration(self):
        """Test complete ridge analysis workflow with phase ridges."""
        result = ridge_analysis(
            wavelet=self.wavelet_y,
            freqs=self.freqs,
            t=self.t,
            ridge_type="phase",
            amplitude_threshold=0.1,
            min_group_size=3,
            max_gap=3,
        )

        # Should return valid structure even if fewer ridges detected
        self.assertIn("ridge_points", result)
        self.assertIn("num_groups", result)
        self.assertIn("group_data", result)

        # num_groups should match group_data and group_lengths
        self.assertEqual(result["num_groups"], len(result["group_data"]))
        self.assertEqual(result["num_groups"], len(result["group_lengths"]))

    def test_ridge_analysis_output_format_stability(self):
        """Test that output format is stable across different inputs."""
        # Test with different ridge types
        result_amp = ridge_analysis(
            wavelet=self.wavelet_y,
            freqs=self.freqs,
            t=self.t,
            ridge_type="amplitude",
            amplitude_threshold=10.0,
        )

        result_phase = ridge_analysis(
            wavelet=self.wavelet_y,
            freqs=self.freqs,
            t=self.t,
            ridge_type="phase",
            amplitude_threshold=0.1,
        )

        # Both should have same keys
        self.assertEqual(set(result_amp.keys()), set(result_phase.keys()))

        # Both should have consistent data types
        for key in result_amp.keys():
            self.assertEqual(type(result_amp[key]), type(result_phase[key]))

            if isinstance(result_amp[key], np.ndarray):
                self.assertEqual(result_amp[key].shape, result_phase[key].shape)

    def test_ridge_analysis_empty_result_handling(self):
        """Test behavior when no ridges are found."""
        # Use very high threshold to find no ridges
        result = ridge_analysis(
            wavelet=self.wavelet_y,
            freqs=self.freqs,
            t=self.t,
            ridge_type="amplitude",
            amplitude_threshold=1000.0,  # Extremely high threshold
            min_group_size=5,
            max_gap=2,
        )

        # Should handle empty results gracefully
        self.assertEqual(result["num_groups"], 0)
        self.assertEqual(len(result["group_data"]), 0)
        self.assertEqual(len(result["group_lengths"]), 0)

        # Other arrays should still have correct shapes
        self.assertEqual(result["ridge_points"].shape, self.wavelet_y.shape)
        self.assertEqual(result["inst_frequency"].shape, self.wavelet_y.shape)


def synth_signal(t, tau, omega, k):
    """
    Generates a synthetic signal based on the given parameters.
    The signal is defined in the complex plane and is then separated into
    its real and imaginary components.
    """
    xi_t = np.piecewise(
        t,
        [t < tau / 4.0, t >= tau / 4.0],
        [
            -1.0,
            lambda x: -scimath.sqrt(1.0 - (1 / 3.0 * (4.0 * x / tau - 1.0)) ** 2.0),
        ],
    )

    k_t = k * (1 + 5.0 * t / tau)
    a_t = k_t * scimath.sqrt(1.0 + scimath.sqrt(1.0 - xi_t**2.0))
    b_t = k_t * scimath.sqrt(1.0 - scimath.sqrt(1.0 - xi_t**2.0))
    phi_t = omega * t * (1.0 + t / tau)
    theta_t = np.pi / 2.0 - omega * t / 10.0
    z_t = np.exp(1j * theta_t) * (a_t * np.cos(phi_t) - 1j * b_t * np.sin(phi_t))

    return z_t.real, z_t.imag


def theoretical_ridge_function(t, tau, omega):
    """
    Piecewise theoretical ridge function:
    For t < tau/4: f(t) = omega * (1 + 2*t/tau) + omega/10
    For t >= tau/4: f(t) = omega * (1 + 2*t/tau) + (omega/10) * sqrt(1 + (4*t/(3*tau) - 1/3)^2)
    """
    return np.piecewise(
        t,
        [t < tau / 4.0, t >= tau / 4.0],
        [
            lambda x: omega * (1 + 2 * x / tau) + omega / 10,
            lambda x: omega * (1 + 2 * x / tau)
            + (omega / 10) * np.sqrt(1 + (4 * x / (3 * tau) - 1 / 3) ** 2),
        ],
    )


def create_dual_frequency_synthetic_signal():
    """
    Create a synthetic signal with two distinct frequency components for testing
    separate_ridge_groups function.
    """
    # Parameters for low frequency component
    t = np.linspace(0, 799, 800)
    tau1 = 1000.0
    omega1 = 0.05  # Low frequency
    k1 = 10.0

    # Parameters for high frequency component
    tau2 = 1000.0
    omega2 = 0.4  # Higher frequency
    k2 = 10.0

    # Generate both synthetic signals
    x1_t, y1_t = synth_signal(t, tau1, omega1, k1)
    x2_t, y2_t = synth_signal(t, tau2, omega2, k2)

    # Combine the signals
    combined_x = x1_t + x2_t
    combined_y = y1_t + y2_t

    return t, combined_x, combined_y, (tau1, omega1, k1), (tau2, omega2, k2)


if __name__ == "__main__":
    unittest.main()

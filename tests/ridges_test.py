import unittest

import numpy as np
from numpy.lib import scimath

from clouddrift.wavelet import (
    morse_wavelet_transform,
    morse_logspace_freq,
)
from clouddrift.ridges import (
    gradient_of_angle,
    instmom_univariate,
    instmom_multivariate,
    isridgepoint,
    ridge_shift_interpolation,
    organize_ridge_points,
    create_3d_deviation_matrix,
    bilateral_minimum_selection,
    separate_ridge_groups,
    calculate_ridge_group_lengths,
    ridge_analysis
)


class TestGradientOfAngle(unittest.TestCase):
    def test_no_wrap(self):
        # Simple linear angles should match np.gradient
        angles = np.array([0.0, 1.0, 2.0, 3.0])
        grad = gradient_of_angle(angles, edge_order=1, axis=0)
        expected = np.gradient(angles)
        self.assertTrue(np.allclose(grad, expected))

    def test_wrap(self):
        # Angles wrapping around ±π should unwrap smoothly
        # Create a linear ramp that crosses the ±π boundary
        linear = np.linspace(-np.pi + 0.5, np.pi + 0.5, 10)
        angles = (linear + np.pi) % (2*np.pi) - np.pi
        # Compare gradient_of_angle to np.gradient after unwrapping
        grad_wrap = gradient_of_angle(angles, axis=0)
        expected = np.gradient(np.unwrap(angles))
        np.testing.assert_allclose(grad_wrap, expected, atol=1e-6)

    def test_edge_order_2_even_length(self):
        # Even-length array with edge_order=2 should match np.gradient after unwrap
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        out = gradient_of_angle(angles, edge_order=2, axis=0)
        expected = np.gradient(np.unwrap(angles), edge_order=2)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_edge_order_2_odd_length(self):
        # Odd-length array with edge_order=2 should also match np.gradient after unwrap
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        out = gradient_of_angle(angles, edge_order=2, axis=0)
        expected = np.gradient(np.unwrap(angles), edge_order=2)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_axis_parameter(self):
        # 2D array: compute along axis=1
        base = np.linspace(0, 3.0, 4)
        arr = np.stack([base, -base], axis=0)  # shape (2,4)
        out = gradient_of_angle(arr, edge_order=1, axis=1)
        for i in range(arr.shape[0]):
            np.testing.assert_allclose(out[i], np.gradient(arr[i]), atol=1e-6)


class TestInstantaneousMoments(unittest.TestCase):
    def test_univariate_simple(self):
        # Test analytic signal with unit amplitude and unit frequency
        t = np.linspace(0, 2 * np.pi, 100)
        dt = t[1] - t[0]  # Time step
        signal = np.exp(1j * t)
        amp, omega, upsilon, xi = instmom_univariate(signal, sample_rate=1/dt, axis=0)
        self.assertTrue(np.allclose(amp, 1.0, atol=1e-6))
        self.assertTrue(np.allclose(omega, 1.0, atol=1e-2))
        self.assertTrue(np.allclose(upsilon, 0.0, atol=1e-2))
        self.assertTrue(np.allclose(xi, 0.0 + 1j * 0.0, atol=1e-2))

    def test_multivariate_simple(self):
        # Two-component signal to test joint moments
        t = np.linspace(0, 2 * np.pi, 100)
        dt = t[1] - t[0]  # Time step
        s1 = np.exp(1j * t)
        s2 = np.exp(1j * (2 * t))
        signals = np.stack([s1, s2], axis=1)  # shape (time, 2)
        amp, omega, upsilon, xi = instmom_multivariate(signals, sample_rate=1.0/dt, time_axis=0, joint_axis=1)
        # Basic shape checks and positivity
        self.assertEqual(amp.shape, t.shape)
        self.assertEqual(omega.shape, t.shape)
        self.assertTrue(np.all(amp > 0))


    def test_univariate_complex_signal(self):
        # Complex signal with known frequency and phase
        t = np.linspace(2.0, 6.0, 100)
        dt = t[1] - t[0]  # Time step
        signal = np.log(t)*np.exp(1j * t)
        amp, omega, upsilon, xi = instmom_univariate(signal, sample_rate=1.0/dt, axis=0)
        amps = [np.log(x) for x in t]  # Amplitude is log(t)
        omegas = [1.0] * len(t)  # Frequency is constant 1.0
        upsilons = [1.0/(x*np.log(x)) for x in t]
        xis = [-np.log(x)/(x*np.log(x))**2.0 for x in t]

        # Test the middle points where numerical derivatices are most accurate
        mid_start = len(t) // 10
        mid_end = -len(t) // 10
        
        self.assertTrue(np.allclose(amp, amps, atol=1e-6))
        self.assertTrue(np.allclose(omega, omegas, atol=1e-6))
        self.assertTrue(np.allclose(upsilon[mid_start:mid_end], upsilons[mid_start:mid_end], atol=1e-3)) 
        self.assertTrue(np.allclose(xi[mid_start:mid_end], xis[mid_start:mid_end], atol=1e-3))

    def test_multivariate_complex_signal(self):
        # Complex multivariate signal with known frequencies
        t = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        s1 = np.log(t) * np.exp(1j * t)
        s2 = np.log(t) * np.exp(1j * (2 * t))
        signals = np.stack([s1, s2], axis=1)
        amp, omega, upsilon, xi = instmom_multivariate(signals, sample_rate=1.0, time_axis=0, joint_axis=1)
        
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
            upsilons_1 = [1.0/(x*np.log(x)) for x in t]
            upsilons_2 = [1.0/(x*np.log(x)) for x in t]
            xis_1 = [-np.log(x)/(x*np.log(x))**2.0 for x in t]
            xis_2 = [-np.log(x)/(x*np.log(x))**2.0 for x in t]
            
            # Test the middle points where numerical derivatices are most accurate
            mid_start = len(t) // 10
            mid_end = -len(t) // 10

            self.assertTrue(np.allclose(amp[:, 0], amps_1, atol=1e-6))
            self.assertTrue(np.allclose(amp[:, 1], amps_2, atol=1e-6))
            self.assertTrue(np.allclose(omega[:, 0], omegas_1, atol=1e-6))
            self.assertTrue(np.allclose(omega[:, 1], omegas_2, atol=1e-6))
            self.assertTrue(np.allclose(upsilon[:, 0][mid_start:mid_end], upsilons_1[mid_start:mid_end], atol=1e-3))
            self.assertTrue(np.allclose(upsilon[:, 1][mid_start:mid_end], upsilons_2[mid_start:mid_end], atol=1e-3))
            self.assertTrue(np.allclose(xi[:, 0][mid_start:mid_end], xis_1[mid_start:mid_end], atol=1e-3))
            self.assertTrue(np.allclose(xi[:, 1][mid_start:mid_end], xis_2[mid_start:mid_end], atol=1e-3))


class TestIsRidgePoint(unittest.TestCase):
    def test_empty_transform(self):
        # Empty input should yield empty ridge points
        wt = np.zeros((0, 0), dtype=np.complex128)
        freqs = np.array([])
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.1, 'amplitude')
        self.assertEqual(rp.size, 0)
        self.assertEqual(rq.size, 0)

    def test_basic_amplitude_ridge_vertical(self):
        # Create a Gaussian peak at center scale (ridge across all times at one scale)
        time_points = 101
        scale_points = 50
        t = np.linspace(-5, 5, time_points)
        freqs = np.linspace(0.1, 2.0, scale_points)
        
        # Create sharp Gaussian amplitude ridge at center SCALE
        center_scale_idx = scale_points // 2
        sigma = 2.0
        
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                # Peak at center scale, constant across time
                scale_distance = abs(j - center_scale_idx)
                amplitude = np.exp(-scale_distance**2 / (2 * sigma**2)) + 0.1
                wt[j, i] = amplitude * np.exp(1j * 0.5)
        
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.05, 'amplitude')
        
        # Ridge should be detected at center scale for all times
        self.assertTrue(rp[center_scale_idx, :].all())  # All times at center scale
        
        # Adjacent scales should not be ridges (strict local maximum)
        if center_scale_idx > 0:
            self.assertFalse(rp[center_scale_idx-1, :].any())  # Scale below center
        if center_scale_idx < scale_points - 1:
            self.assertFalse(rp[center_scale_idx+1, :].any())  # Scale above center

    def test_diagonal_amplitude_ridge(self):
        # Create a Gaussian that moves linearly with time (diagonal ridge)
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
                amplitude = np.exp(-scale_distance**2 / (2 * sigma**2)) + 0.1
                wt[j, i] = amplitude * np.exp(1j * 0.5)
        
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.2, 'amplitude')
        
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
        # Test that amplitude threshold properly filters weak signals
        time_points = 51
        scale_points = 25
        t = np.linspace(-3, 3, time_points)
        freqs = np.linspace(0.1, 1.0, scale_points)
        
        # Create weak amplitude ridge at center scale
        center_scale_idx = scale_points // 2
        sigma = 2.0
        
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                # Peak at center scale with max amplitude = 0.5
                scale_distance = abs(j - center_scale_idx)
                amplitude = 0.5 * np.exp(-scale_distance**2 / (2 * sigma**2)) + 0.1
                wt[j, i] = amplitude * np.exp(1j * 0.0)
        
        # High threshold should reject ridge
        rp_high, _, _, _ = isridgepoint(wt, freqs, 0.8, 'amplitude')
        self.assertFalse(rp_high.any())
        
        # Low threshold should detect ridge
        rp_low, _, _, _ = isridgepoint(wt, freqs, 0.2, 'amplitude')
        self.assertTrue(rp_low[center_scale_idx, :].all())  # Center scale detected


    def test_phase_ridge_linear_chirp_increasing(self):
        # Test phase ridge detection with linear chirp
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
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.01, 'phase')

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
        for i, (ridge_time, ridge_freq) in enumerate(zip(ridge_time_vals_filtered, ridge_freqs_filtered)):
            theoretical_time = ridge_freq / (2 * chirp_rate)
            time_error = abs(ridge_time - theoretical_time)
            self.assertLess(time_error, 0.5001)
    
    def test_phase_ridge_linear_chirp_decreasing(self):
        # Test phase ridge detection with linear chirp and decreasing frequency matrix
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
        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.01, 'phase')

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
        for i, (ridge_time, ridge_freq) in enumerate(zip(ridge_time_vals_filtered, ridge_freqs_filtered)):
            theoretical_time = ridge_freq / (2 * chirp_rate)
            time_error = abs(ridge_time - theoretical_time)
            print("Ridge time:", ridge_time, "Theoretical time:", theoretical_time, "Error:", time_error)
            self.assertLess(time_error, 0.5001)

    def test_phase_ridge_non_monotonic_frequency_error(self):
        # Test that non-monotonic frequency matrix raises an error for phase ridges
        time_points = 50
        scale_points = 25

        # Use unit time spacing as assumed by isridgepoint
        t = np.arange(time_points, dtype=float)  # 0, 1, 2, ..., 49
        
        # Create non-monotonic frequencies (goes up then down)
        freqs = np.concatenate([
            np.linspace(0.1, 0.6, scale_points // 2),
            np.linspace(0.5, 0.1, scale_points - scale_points // 2)
        ])

        # Create a simple linear chirp with quadratic phase
        chirp_rate = 0.02  # Hz per second
        phase = chirp_rate * t**2  # Quadratic phase, no scale dependence
        
        # Create wavelet transform with same phase for all scales
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for j in range(scale_points):
            wt[j, :] = 1.0 * np.exp(1j * phase)

        # Should raise ValueError for non-monotonic frequency matrix in phase ridge detection
        with self.assertRaises(ValueError) as context:
            rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.01, 'phase')
        
        # Check that the error message mentions monotonic frequency requirement
        self.assertIn("monotonic", str(context.exception).lower())

    def test_frequency_constraints(self):
        # Test frequency min/max constraints
        time_points = 51
        scale_points = 25
        t = np.linspace(-3, 3, time_points)
        freqs = np.linspace(0.1, 2.0, scale_points)
        
        # Create amplitude ridge at center
        sigma = 0.5
        amplitudes = np.exp(-t**2 / (2 * sigma**2)) + 0.1
        
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                wt[j, i] = amplitudes[i] * np.exp(1j * j)  # varying phase
        
        # Constrain to middle frequency range
        freq_min = 0.8
        freq_max = 1.2
        
        rp, rq, proc, inst_freq = isridgepoint(
            wt, freqs, 0.5, 'amplitude', freq_min=freq_min, freq_max=freq_max
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
        # Create a Gaussian that moves linearly with time (diagonal ridge)
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
                amplitude = np.exp(-scale_distance**2 / (2 * sigma**2)) + 0.1
                wt[j, i, 0] = amplitude * np.exp(1j * 0.5)
                wt[j, i, 1] = amplitude * np.exp(1j * 1.0)

        rp, rq, proc, inst_freq = isridgepoint(wt, freqs, 0.1, 'amplitude')

        # Debug: Check what we actually created
        center_idx = time_points // 2
        
        # Should detect ridge at center time - ridge is along scale axis (axis 0)
        self.assertTrue(rp[:, center_idx].any())

        # Output shapes should match 2D case
        self.assertEqual(rp.shape, (scale_points, time_points))
        self.assertEqual(rq.shape, (scale_points, time_points))

    def test_ridge_quantity_values(self):
        # Test that ridge quantities have expected values
        time_points = 31
        scale_points = 15
        t = np.linspace(-2, 2, time_points)
        freqs = np.linspace(0.5, 1.5, scale_points)
        
        # Create simple amplitude ridge
        sigma = 0.4
        amplitudes = np.exp(-t**2 / (2 * sigma**2)) + 0.1
        
        wt = np.zeros((scale_points, time_points), dtype=np.complex128)
        for i in range(time_points):
            for j in range(scale_points):
                wt[j, i] = amplitudes[i] * np.exp(1j * 0.5)
        
        # Test amplitude ridge
        rp_amp, rq_amp, _, _ = isridgepoint(wt, freqs, 0.1, 'amplitude')
        
        # Ridge quantity for amplitude should be the amplitude itself
        amp, _, _, _ = instmom_univariate(wt, axis=1)
        self.assertTrue(np.allclose(rq_amp, amp))
        
        # Test phase ridge
        rp_phase, rq_phase, _, inst_freq = isridgepoint(wt, freqs, 0.1, 'phase')
        
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
        int_data = [np.ones((10, 10), dtype=np.int32), np.ones((10, 10), dtype=np.int32)]
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
        rp, rq, proc, inst_freq = isridgepoint(wavelet_y, freqs, amplitude_threshold=10, ridge_type='amplitude')
        
        # Create frequency meshgrid and power
        freq_mesh = np.zeros_like(rq)
        for i, f in enumerate(freqs):
            freq_mesh[i, :] = f
        power = np.abs(wavelet_y)**2
        
        # Test ridge_shift_interpolation
        result = ridge_shift_interpolation(rp, rq, [power, freq_mesh])
        
        # Basic functionality tests
        assert len(result) == 2
        
        # Get the shifted ridge point frequencies
        interpolated_ridge_freqs = result[1]
        
        # Accuracy test against theoretical ridge
        ridge_scales, ridge_times = np.where(rp)
        ridge_time_vals = t[ridge_times]
        theoretical_at_ridge_times = theoretical_ridge_function(ridge_time_vals, tau, omega)
        
        # Calculate RMS error
        rms_error = np.sqrt(np.mean((interpolated_ridge_freqs - theoretical_at_ridge_times)**2))
        assert rms_error < 0.005
        
        # Check frequency range alignment
        theoretical_full = theoretical_ridge_function(t, tau, omega)
        theoretical_range = theoretical_full.max() - theoretical_full.min()
        interpolated_range = interpolated_ridge_freqs.max() - interpolated_ridge_freqs.min()
        
        # Ranges should be similar (within 20%)
        range_ratio = interpolated_range / theoretical_range
        assert 0.8 <= range_ratio <= 1.2
        
        # Data consistency test
        assert len(result[0]) == len(result[1])
        
        # All values should be finite
        for res in result:
            assert np.all(np.isfinite(res))


class TestOrganizeRidgePoints(unittest.TestCase):
    def test_skip(self):
        self.skipTest('Not yet implemented')


class TestCreate3DDeviationMatrix(unittest.TestCase):
    def test_skip(self):
        self.skipTest('Not yet implemented')


class TestBilateralMinimumSelection(unittest.TestCase):
    def test_skip(self):
        self.skipTest('Not yet implemented')


class TestSeparateRidgeGroups(unittest.TestCase):
    def test_skip(self):
        self.skipTest('Not yet implemented')


class TestCalculateRidgeGroupLengths(unittest.TestCase):
    def test_skip(self):
        self.skipTest('Not yet implemented')


class TestRidgeAnalysis(unittest.TestCase):
    def test_skip(self):
        self.skipTest('Not yet implemented')


def synth_signal(t, tau, omega, k):
    """
    Generates a synthetic signal based on the given parameters.
    The signal is defined in the complex plane and is then separated into
    its real and imaginary components.
    """
    xi_t = np.piecewise(
        t,
        [t < tau/4.0, t >= tau/4.0],
        [
            -1.0,
            lambda x: -scimath.sqrt(1.0 - (1/3.0*(4.0 * x / tau - 1.0)) ** 2.0),
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
        [t < tau/4.0, t >= tau/4.0],
        [
            lambda x: omega * (1 + 2*x/tau) + omega/10,
            lambda x: omega * (1 + 2*x/tau) + (omega/10) * np.sqrt(1 + (4*x/(3*tau) - 1/3)**2)
        ]
    )


if __name__ == "__main__":
    unittest.main()
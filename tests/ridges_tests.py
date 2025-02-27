import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import sys
import os

# Add the parent directory to the path to import the function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clouddrift.ridges import ridgemap  # Adjust import as needed


class TestRidgeMap(unittest.TestCase):
    
    def test_basic_1d_single_quantity(self):
        """Test mapping a single ridge quantity to a 1D time series."""
        # Create sample data: a ridge quantity and corresponding indices
        xr = np.array([10, 20, 30, 40, 50])
        ir = np.array([1, 2, 3, 4, 5])  # 1-indexed positions
        
        # Map the ridge quantity to a time series of length 10
        result = ridgemap(10, xr, ir)
        
        # Expected: mapped quantity and multiplicity
        expected_x = np.array([[10], [20], [30], [40], [50], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]])
        expected_mult = np.array([[1], [1], [1], [1], [1], [0], [0], [0], [0], [0]])
        
        # Check results
        assert_array_equal(np.isnan(result[0]), np.isnan(expected_x))
        assert_array_almost_equal(result[0][~np.isnan(result[0])], 
                                 expected_x[~np.isnan(expected_x)])
        assert_array_equal(result[1], expected_mult)
    
    def test_basic_1d_multiple_quantities(self):
        """Test mapping multiple ridge quantities to a 1D time series."""
        # Create sample data: two ridge quantities and corresponding indices
        xr = np.array([10, 20, 30])
        fr = np.array([1.1, 2.2, 3.3])
        ir = np.array([1, 3, 5])  # 1-indexed positions
        
        # Map the ridge quantities to a time series of length 6
        result = ridgemap(6, xr, fr, ir)
        
        # Expected mapped quantities and multiplicity
        expected_x = np.array([[10], [np.nan], [20], [np.nan], [30], [np.nan]])
        expected_f = np.array([[1.1], [np.nan], [2.2], [np.nan], [3.3], [np.nan]])
        expected_mult = np.array([[1], [0], [1], [0], [1], [0]])
        
        # Check results
        self.assertEqual(len(result), 3)  # x, f, and multiplicity
        
        assert_array_equal(np.isnan(result[0]), np.isnan(expected_x))
        assert_array_almost_equal(result[0][~np.isnan(result[0])], 
                                 expected_x[~np.isnan(expected_x)])
                                 
        assert_array_equal(np.isnan(result[1]), np.isnan(expected_f))
        assert_array_almost_equal(result[1][~np.isnan(result[1])], 
                                 expected_f[~np.isnan(expected_f)])
                                 
        assert_array_equal(result[2], expected_mult)
    
    def test_2d_grid(self):
        """Test mapping ridge quantities to a 2D grid."""
        # Create sample data for 2D grid mapping
        xr = np.array([100, 200, 300])
        ir = np.array([1, 2, 3])  # Row indices (1-indexed)
        kr = np.array([2, 3, 1])  # Column indices (1-indexed)
        
        # Map the ridge quantity to a 4x4 grid
        result = ridgemap((4, 4), xr, ir, kr)
        
        # Expected: 2D mapped quantity and multiplicity
        expected_x = np.full((4, 4), np.nan)
        expected_x[0, 1] = 100  # (ir=1, kr=2) -> (0-indexed: 0, 1)
        expected_x[1, 2] = 200  # (ir=2, kr=3) -> (0-indexed: 1, 2)
        expected_x[2, 0] = 300  # (ir=3, kr=1) -> (0-indexed: 2, 0)
        
        expected_mult = np.zeros((4, 1))
        expected_mult[0, 0] = 1
        expected_mult[1, 0] = 1
        expected_mult[2, 0] = 1
        
        # Check results
        assert_array_equal(np.isnan(result[0]), np.isnan(expected_x))
        assert_array_almost_equal(result[0][~np.isnan(result[0])], 
                                expected_x[~np.isnan(expected_x)])
        assert_array_equal(result[1], expected_mult)
    
    def test_multiple_ridges_with_nans(self):
        """Test mapping multiple ridges separated by NaNs."""
        # Create sample data with two ridges separated by NaNs
        xr = np.array([[10, 20, np.nan, 40, 50],   # First ridge
                       [15, 25, 35, np.nan, 55]])  # Second ridge
        ir = np.array([[1, 2, np.nan, 4, 5],       # Indices for first ridge
                       [1, 2, 3, np.nan, 5]])      # Indices for second ridge
        
        # Map the ridge quantities to a time series of length 6
        result = ridgemap(6, xr, ir)
        
        # Expected: mapped quantity with separate columns for each ridge
        expected_x = np.full((6, 2), np.nan)
        expected_x[0, 0] = 10  # First ridge, first position
        expected_x[1, 0] = 20  # First ridge, second position
        expected_x[3, 0] = 40  # First ridge, fourth position
        expected_x[4, 0] = 50  # First ridge, fifth position
        
        expected_x[0, 1] = 15  # Second ridge, first position
        expected_x[1, 1] = 25  # Second ridge, second position
        expected_x[2, 1] = 35  # Second ridge, third position
        expected_x[4, 1] = 55  # Second ridge, fifth position
        
        expected_mult = np.array([[2], [2], [1], [1], [2], [0]])
        
        # Check results
        assert_array_equal(np.isnan(result[0]), np.isnan(expected_x))
        assert_array_almost_equal(result[0][~np.isnan(result[0])], 
                                expected_x[~np.isnan(expected_x)])
        assert_array_equal(result[1], expected_mult)
    
    def test_collapse_functionality(self):
        """Test collapsing multiple ridges into a single time series."""
        # Create sample data with two ridges
        xr = np.array([[10, 20, 30],    # First ridge
                       [15, 25, 35]])   # Second ridge
        fr = np.array([[1.0, 2.0, 3.0], # First ridge frequencies
                       [1.5, 2.5, 3.5]]) # Second ridge frequencies
        ir = np.array([[1, 2, 3],       # Indices for first ridge
                       [1, 2, 3]])      # Indices for second ridge
        
        # Map with collapse=True
        result = ridgemap(4, xr, fr, ir, collapse=True)
        
        # Expected: collapsed quantities (sum for first quantity, average for others)
        # For xr: sum of values at each position
        # For fr: average of values at each position
        expected_x = np.array([[25], [45], [65], [0]])  # Sum of xr values
        expected_mult = np.array([[2], [2], [2], [0]])  # Multiplicity
        
        # Check results
        self.assertEqual(len(result), 2)  # Only one quantity plus multiplicity when collapsed
        assert_array_almost_equal(result[0], expected_x)
        assert_array_equal(result[1], expected_mult)
    
    def test_empty_input(self):
        """Test handling of empty/None inputs."""
        # Test with None indices
        result = ridgemap(5, np.array([1, 2, 3]), None)
        
        expected_x = np.full((5, 1), np.nan)
        expected_mult = np.zeros((5, 1))
        
        assert_array_equal(np.isnan(result[0]), np.isnan(expected_x))
        assert_array_equal(result[1], expected_mult)
        
        # Test with empty indices
        result = ridgemap(5, np.array([1, 2, 3]), np.array([]))
        
        assert_array_equal(np.isnan(result[0]), np.isnan(expected_x))
        assert_array_equal(result[1], expected_mult)
    
    def test_multiplicity_calculation(self):
        """Test proper calculation of ridge multiplicity."""
        # Create data with varying ridge presence
        xr1 = np.array([10, 20, 30])
        ir1 = np.array([1, 2, 3])
        
        xr2 = np.array([15, 25])
        ir2 = np.array([1, 3])
        
        xr3 = np.array([17])
        ir3 = np.array([1])
        
        # Combine into 2D arrays
        xr = np.column_stack([xr1, xr2, xr3])
        ir = np.column_stack([ir1, ir2, ir3])
        
        # Add NaN padding
        xr = np.pad(xr, ((0, 0), (0, 1)), mode='constant', constant_values=np.nan)
        ir = np.pad(ir, ((0, 0), (0, 1)), mode='constant', constant_values=np.nan)
        
        # Map to a time series of length 5
        result = ridgemap(5, xr, ir)
        
        # Expected multiplicity: position 1 has 3 ridges, position 2 has 1, position 3 has 2
        expected_mult = np.array([[3], [1], [2], [0], [0]])
        
        # Check multiplicity calculation
        assert_array_equal(result[1], expected_mult)

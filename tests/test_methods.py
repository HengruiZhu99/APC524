import unittest
import numpy as np
from navier_stokes.fdm_fvm_functions import *


class TestNumericalMethods(unittest.TestCase):
    """
    Unit tests for Finite Difference Method (FDM) and Finite Volume Method (FVM).
    """

    def setUp(self):
        """
        Setup common parameters for the tests.
        """
        self.dx = 1.0
        self.nu = 0.1
        self.fdm_method = FDMMethod()
        self.fvm_method = FVMMethod()

    def compute_expected_rhs_fdm(self, u):
        """
        Dynamically compute the expected RHS for the FDM method.
        """
        rhs = -0.5 * (u * (np.roll(u, -1) - np.roll(u, +1))) / self.dx
        rhs += self.nu * (np.roll(u, -1) - 2 * u + np.roll(u, +1)) / self.dx**2
        return apply_boundary_conditions(rhs)

    def compute_expected_rhs_fvm(self, u):
        """
        Dynamically compute the expected RHS for the FVM method.
        """
        rhs = -((np.roll(u, -1) ** 2 - np.roll(u, +1) ** 2)) / (2 * self.dx)
        rhs += self.nu * (np.roll(u, -1) - 2 * u + np.roll(u, +1)) / self.dx**2
        return apply_boundary_conditions(rhs)

    def test_fdm_varied_values(self):
        """
        Test FDM method with varied inputs: positive, negative, and mixed values.
        """
        test_cases = [
            np.array([1, 2, 3, 4, 5], dtype=float),  
            np.array([-1, -2, -3, -4, -5], dtype=float),  
            np.array([-1, 2, -3, 4, -5], dtype=float),  
        ]
        for u in test_cases:
            with self.subTest(u=u):
                rhs = self.fdm_method(u, self.dx, self.nu)
                expected_rhs = self.compute_expected_rhs_fdm(u)
                np.testing.assert_almost_equal(rhs, expected_rhs, decimal=3)

    def test_fvm_varied_values(self):
        """
        Test FVM method with varied inputs: positive, negative, and mixed values.
        """
        test_cases = [
            np.array([1, 2, 3, 4, 5], dtype=float),  # Positive values
            np.array([-1, -2, -3, -4, -5], dtype=float),  # Negative values
            np.array([-1, 2, -3, 4, -5], dtype=float),  # Mixed values
        ]
        for u in test_cases:
            with self.subTest(u=u):
                rhs = self.fvm_method(u, self.dx, self.nu)
                expected_rhs = self.compute_expected_rhs_fvm(u)
                np.testing.assert_almost_equal(rhs, expected_rhs, decimal=5)
                
if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from navier_stokes.fdm_fvm_functions import *

class TestComputeSecondDerivative(unittest.TestCase):
    """
    Unit test for the compute_second_derivative function for smaller dx.
    """

    def test_compute_second_derivative(self):
        """
        Test second derivative calculation using a known analytic function.
        """
        dx = 0.001  # Small grid spacing
        x = np.linspace(0, 1, int(1 / dx) + 1)  # Uniform grid
        u = np.sin(2 * np.pi * x)  # Test function

        # Analytic second derivative
        expected_u_xx = -(2 * np.pi) ** 2 * np.sin(2 * np.pi * x)

        # Numerical second derivative
        u_xx = compute_second_derivative(u, dx)

        # Test with relaxed precision
        np.testing.assert_almost_equal(u_xx[1:-1], expected_u_xx[1:-1], decimal=2)

        # Validate boundary conditions
        self.assertEqual(u_xx[0], u_xx[-2])  # Periodic: First = Second-to-last
        self.assertEqual(u_xx[-1], u_xx[1])  # Periodic: Last = Second


if __name__ == "__main__":
    unittest.main()

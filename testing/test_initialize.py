import unittest
import numpy as np
from navier_stokes.fdm_fvm_functions import *

class TestInitialize(unittest.TestCase):
    """
    Unit tests for the initialization functions: Gaussian and Sinusoidal.
    """

    def test_initialize_gaussian(self):
        """
        Test GaussionInitialize to ensure:
        Correct grid points (x) are computed.
        Initial Gaussian condition is generated.
        dx matches L / (nx - 1).
        """
        nx = 10  # Use a larger grid for better accuracy
        L = 1.0
        initializer = GaussionInitialize()  # Use the class, not a function
        x, u_initial, dx = initializer.initialize(nx, L)

        # Verify length of arrays
        self.assertEqual(len(x), nx)
        self.assertEqual(len(u_initial), nx)

        # Verify dx calculation
        self.assertAlmostEqual(dx, L / (nx - 1))

        # Verify Gaussian initial condition
        peak_index = np.argmin(np.abs(x - 0.5 * L))
        expected_peak = np.exp(-100 * (x[peak_index] - 0.5 * L) ** 2)
        self.assertAlmostEqual(u_initial[peak_index], expected_peak, places=7)

    def test_initialize_sinusoidal(self):
        """
        Test SinusoidalInitialize to ensure:
        Correct grid points (x) are computed.
        Initial sinusoidal condition is generated.
        dx matches L / (nx - 1).
        """
        nx = 5
        L = 1.0
        initializer = SinusoidalInitialize()  # Use the class, not a function
        x, u_initial, dx = initializer.initialize(nx, L)

        # Verify length of arrays
        self.assertEqual(len(x), nx)
        self.assertEqual(len(u_initial), nx)

        # Verify dx calculation
        self.assertAlmostEqual(dx, L / (nx - 1))

        # Verify sinusoidal condition
        expected_first = np.sin(2 * np.pi * x[0] / L)
        self.assertAlmostEqual(u_initial[0], expected_first, places=7)

if __name__ == "__main__":
    unittest.main()

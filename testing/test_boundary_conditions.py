import unittest
import numpy as np
from main_code import apply_boundary_conditions

class TestBoundaryConditions(unittest.TestCase):
    """
    Unit tests for the apply_boundary_conditions function.
    """

    def test_apply_boundary_conditions(self):
        """
        Test that periodic boundary conditions are correctly applied.
        """
        u = np.array([1, 2, 3, 6, 5])
        u_updated = apply_boundary_conditions(u.copy())

        # Verify periodic boundary conditions
        self.assertEqual(u_updated[0], u_updated[-2])
        self.assertEqual(u_updated[-1], u_updated[1])

if __name__ == "__main__":
    unittest.main()




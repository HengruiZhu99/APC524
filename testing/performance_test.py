import time
import numpy as np
from navier_stokes.fdm_fvm_functions import *


def performance_test():
    """
    Measure the performance of various configurations.
    """
    # Define test configurations
    configurations = [
        {"nx": 200, "nt": 10, "L": 1.0, "integrator": EulerIntegrator()},
        {"nx": 500, "nt": 50, "L": 1.0, "integrator": RK2Integrator()},
        {"nx": 1000, "nt": 100, "L": 1.0, "integrator": RK4Integrator()},
    ]

    results = []

    for config in configurations:
        # Extract configuration parameters
        nx = config["nx"]
        nt = config["nt"]
        L = config["L"]
        integrator = config["integrator"]

        # Set up Gaussian initializer and FDM method
        initializer = GaussionInitialize()
        method = FDMMethod()
        nu = 0.1  # Viscosity

        # Measure execution time
        start_time = time.time()
        integrator.integrate(
            initializer=initializer,
            method=method,
            nx=nx,
            L=L,
            nu=nu,
            nt=nt,
        )
        end_time = time.time()

        execution_time = end_time - start_time
        results.append(
            {
                "nx": nx,
                "nt": nt,
                "integrator": integrator.__class__.__name__,
                "time": execution_time,
            }
        )

    # Print results
    print("Performance Test Results")
    print("=" * 40)
    for result in results:
        print(
            f"Grid Size: {result['nx']:4}, Time Steps: {result['nt']:3}, "
            f"Integrator: {result['integrator']:15}, Time: {result['time']:.4f} seconds"
        )


if __name__ == "__main__":
    performance_test()

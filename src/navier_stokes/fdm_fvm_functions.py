from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
import abc

# Handy type alias
FArray = NDArray[np.float64]


###################
# Initialization #
###################


class Initialize(abc.ABC):
    """
    Initialize the domain and initial condition
    for solving the 1D Navier Stokes equation.
    """

    @abc.abstractmethod
    def initialize(nx: int, L: float) -> tuple[FArray, FArray, float]:
        """
        Args:
            nx (int) = number of points in space
            L (float) = length of domain

        Returns:
            (FArray) = position
            (FArray) = initial fluid velocity
            (float) = length of step in space
        """
        pass


class GaussionInitialize(Initialize):
    """
    Initializes the domain and initial conditions.
    Initial fluid velocity follows a gaussian distribution.
    """

    def initialize(self, nx: int, L: float) -> tuple[FArray, FArray, float]:
        """
        Args:
            nx (int) = number of points in space
            L (float) = length of domain

        Returns:
            x (FArray) = position
            u_initial (FArray) = initial fluid velocity
            dx (float) = length of step in space
        """
        dx = L / (nx - 1)
        x = np.linspace(0, L, nx)
        u_initial = np.exp(-100 * (x - 0.5 * L) ** 2)
        return x, u_initial, dx


class SinusoidalInitialize(Initialize):
    """
    Initializes the domain and initial conditions.
    Initial fluid velocity set as a sinusoidal curve.
    """

    def initialize(self, nx: int, L: float) -> tuple[FArray, FArray, float]:
        """
        Args:
            nx (int) = number of points in space
            L (float) = length of domain

        Returns:
            x (FArray) = position
            u_initial (FArray) = initial fluid velocity
            dx (float) = length of step in space
        """
        dx = L / (nx - 1)
        x = np.linspace(0, L, nx)
        u_initial = np.sin(2 * np.pi * x / L)
        return x, u_initial, dx


#########################
## Auxiliary functions ##
#########################


def apply_boundary_conditions(u: FArray) -> FArray:
    """
    Applies periodic boundary condition

    Args:
        u (FArray) = fluid velocity

    Returns:
        u (FArray) = fluid velocity with boundary conditions applied at first and last step
    """
    u[0] = u[-2]
    u[-1] = u[1]
    return u


def compute_second_derivative(u: FArray, dx: float) -> FArray:
    """
    Calculate viscous term, the second derivative of the velocity with respect to position,
    using the central differential method.

    Args:
        u (FArray) = fluid velocity
        dx (float) = length of step in space

    Returns:
        u_xx (FArray) = viscous term
    """
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    u_xx = apply_boundary_conditions(u_xx)
    return u_xx


#########################
## Numerical Methods ##
#########################

class NumericalMethod(abc.ABC):
    """
    Numerical Methods for spatially discretizing the 1D Navier Stokes equation.
    """

    @abc.abstractmethod
    def __call__(self, u: FArray, dx: float, nu: float) -> FArray:
        """
        Args:
            u (FArray) = fluid velocity
            dx (float) = length of step in space
            nu (float) = viscocity

        Returns:
            (FArray) = discretized navier stokes equation
        """
        pass


class FDMMethod(NumericalMethod):
    """
    Finite difference method for spatially discretizing the 1D Navier Stokes equation.
    """

    def __call__(self, u: FArray, dx: float, nu: float) -> FArray:
        """
        Args:
            u (FArray) = fluid velocity
            dx (float) = Length of step in space
            nu (float) = viscocity

        Returns:
            rhs (FArray) = discretized navier stokes equation
        """
        rhs = np.zeros_like(u)
        rhs = -0.5 * (u * (np.roll(u, -1) - np.roll(u, +1))) / dx
        rhs += nu * compute_second_derivative(u, dx)
        rhs = apply_boundary_conditions(rhs)
        return rhs


class FVMMethod(NumericalMethod):
    """
    Finite volume method for spatially discretizing the 1D Navier Stokes equation.
    """

    def __call__(self, u: FArray, dx: float, nu: float) -> FArray:
        """
        Args:
            u (FArray) = fluid velocity
            dx (float) = Length of step in space
            nu (float) = viscocity

        Returns:
            rhs (FArray) = discretized navier stokes equation
        """
        rhs = np.zeros_like(u)
        rhs = -((np.roll(u, -1)) ** 2 - (np.roll(u, +1)) ** 2) / (2 * dx)
        rhs += nu * compute_second_derivative(u, dx)
        rhs = apply_boundary_conditions(rhs)
        return rhs


######################
## Time Integrators ##
######################


class IntegratorBase(abc.ABC):
    """
    Integrates a 1D Navier Stokes equation in time
    """

    @abc.abstractmethod
    def compute_step(
        self,
        method: NumericalMethod,
        nu: float,
        u: FArray,
        dx: float,
    ) -> FArray:
        """
        Args:
            method (NumericalMethod) = method for spatially discretizing equation
            nu (float) = viscocity
            u (FArray) = fluid velocity
            dx (float) = length of step in space

        Returns:
            (FArray) = fluid velocity at the next time step
        """
        pass

    def integrate(self, initializer: Initialize, method: NumericalMethod, nx: int, L: float, nu: float, nt: int,) -> FArray:
        """
        Args:
            initializer (Initialize) = method initializing equation
            method (NumericalMethod) = method for spatially discretizing equation
            nx (int) = number of spatial points
            L (float) = length of domain
            nu (float) = viscocity
            nt (int) = number of steps in time
        
        Returns:
            u (FArray) = fluid velocity integrated in time
        """
        x, u_initial, dx = initializer.initialize(nx, L)
        u = u_initial.copy()

        for _ in range(nt):
            u = self.compute_step(method, nu, u, dx)
       
        return u


class EulerIntegrator(IntegratorBase):
    """
    Uses the Euler method of integration, using current position to find the next.
    """
    def compute_step(self, method: NumericalMethod, nu: float, u: FArray, dx: float,)-> FArray:
        """
        Args:
            method (NumericalMethod) = method for spatially discretizing equation
            nu (float) = viscocity
            u (FArray) = fluid velocity
            dx (float) = length of step in space
        
        Returns:
            u (FArray) = fluid velocity at the next time step
        """
        dydt = method(u, dx, nu)
        u += dydt * dx
        u = apply_boundary_conditions(u)
        return u

class RK2Integrator(IntegratorBase):
    """
    Uses second-order Runge-Kutta (RK2) method to compute integration in time
    """
    def compute_step(self, method: NumericalMethod, nu: float, u: FArray, dx: float,)-> FArray:
        """
        Args:
            method (NumericalMethod) = method for spatially discretizing equation
            nu (float) = viscocity
            u (FArray) = fluid velocity
            dx (float) = length of step in space
        
        Returns:
            result (FArray) = fluid velocity at the next time step
        """
        # Compute k1 through k2
        k1 = dx * method(u, dx, nu)
        k2 = dx * method(u + k1, dx, nu)

        result: FArray = u + 0.5 * (k1 + k2)
        return result

class RK4Integrator(IntegratorBase):
    """
    Uses fourth-order Runge-Kutta (RK4) method to compute integration in time
    """
    def compute_step(self, method: NumericalMethod, nu: float, u: FArray, dx: float,)-> FArray:
        """
        Args:
            method (NumericalMethod) = method for spatially discretizing equation
            nu (float) = viscocity
            u (FArray) = fluid velocity
            dx (float) = length of step in space
        
        Returns:
            result (FArray) = fluid velocity at the next time step
        """
        # Compute k1 through k4
        k1 = dx * method(u, dx, nu)
        k2 = dx * method(u + k1 / 2, dx, nu)
        k3 = dx * method(u + k2 / 2, dx, nu)
        k4 = dx * method(u + k3, dx, nu)

        # Specifying static type here because mypy + numpy
        # loses track over multiple operations sometimes
        result: FArray = u + (1 / 6) * (k1 + 2. * k2 + 2. * k3 + k4)
        return result

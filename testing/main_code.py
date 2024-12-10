from __future__ import annotations  
from collections.abc import Callable
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
import abc

# Handy type alias
FArray = NDArray[np.float64]

# Function
## Initialize the domain and initial condition.
class Initialize(abc.ABC):
    @abc.abstractmethod
    def initialize(self, nx: int, L: float) -> tuple[FArray, FArray, float]:
        pass

class GaussionInitialize(Initialize):
    def initialize(self, nx: int, L: float) -> tuple[FArray, FArray, float]:
        dx = L / (nx - 1)
        x = np.linspace(0, L, nx)
        u_initial = np.exp(-100 * (x - 0.5 * L) ** 2)
        return x, u_initial, dx
    
class SinusoidalInitialize(Initialize):
    def initialize(self, nx: int, L: float) -> tuple[FArray, FArray, float]:
        dx = L / (nx - 1)
        x = np.linspace(0, L, nx)
        u_initial = np.sin(2 * np.pi * x / L)
        return x, u_initial, dx

## Apply boundary conditions with periodic boundary condition. 
def apply_boundary_conditions(u: FArray) -> FArray:
    u[0] = u[-2]
    u[-1] = u[1]
    return u

## Calculate viscous term with central differential method
def compute_second_derivative(u: FArray, dx: float) -> FArray:
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
    u_xx = apply_boundary_conditions(u_xx)
    return u_xx

# Numerical methods class
class NumericalMethod(abc.ABC):
    @abc.abstractmethod
    def __call__(self, u: FArray, dx: float, nu: float) -> FArray:
        pass

# Finite Difference Method class
class FDMMethod(NumericalMethod):
    def __call__(self, u: FArray, dx: float, nu: float) -> FArray:
        rhs = np.zeros_like(u)
        rhs = -0.5 * (u * (np.roll(u, -1) - np.roll(u, +1))) / dx
        rhs += nu * compute_second_derivative(u, dx)
        rhs = apply_boundary_conditions(rhs)
        return rhs

# Finite Volume Method class
class FVMMethod(NumericalMethod):
    def __call__(self, u: FArray, dx: float, nu: float) -> FArray:
        rhs = np.zeros_like(u)
        rhs = -((np.roll(u, -1))**2 - (np.roll(u, +1))**2) / (2 * dx)
        rhs += nu * compute_second_derivative(u, dx)
        rhs = apply_boundary_conditions(rhs)
        return rhs

# Time integrator class
class IntegratorBase(abc.ABC):
    @abc.abstractmethod
    def compute_step(self, method: NumericalMethod, t_n: float, y_n: FArray, h: float) -> FArray:
        pass

    def integrate(self, initializer: Initialize, method: NumericalMethod, nx: int, L: float, nu: float, nt: int) -> FArray:
        x, u_initial, dx = initializer.initialize(nx, L)
        u = u_initial.copy()

        for _ in range(nt):
            u = self.compute_step(method, nu, u, dx)
       
        return u


class EulerIntegrator(IntegratorBase):
    def compute_step(self, method: NumericalMethod, nu: float, u: FArray, dx: float) -> FArray:
        dydt = method(u, dx, nu)
        u += dydt * dx
        u = apply_boundary_conditions(u)
        return u

class RK2Integrator(IntegratorBase):
    def compute_step(self, method: NumericalMethod, nu: float, u: FArray, dx: float) -> FArray:
        k1 = dx * method(u, dx, nu)
        k2 = dx * method(u + k1, dx, nu)

        result: FArray = u + 0.5 * (k1 + k2)
        return result

class RK4Integrator(IntegratorBase):
    def compute_step(self, method: NumericalMethod, nu: float, u: FArray, dx: float) -> FArray:
        k1 = dx * method(u, dx, nu)
        k2 = dx * method(u + k1 / 2, dx, nu)
        k3 = dx * method(u + k2 / 2, dx, nu)
        k4 = dx * method(u + k3, dx, nu)

        result: FArray = u + (1 / 6) * (k1 + 2. * k2 + 2. * k3 + k4)
        return result

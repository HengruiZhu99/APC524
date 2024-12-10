import pytest
import numpy as np
from FD import (
    initialize, apply_boundary_conditions, compute_second_derivative,
    cal_rhs, rk2_step, run_simulation
)

def test_initialize():
    nx, L = 100, 1.0
    x, u_initial, dx = initialize(nx, L)
    assert len(x) == nx
    assert len(u_initial) == nx
    assert np.isclose(dx, L / (nx - 1))
    assert np.all(u_initial >= 0) and np.all(u_initial <= 1)

def test_apply_boundary_conditions():
    u = np.array([1, 2, 3, 4, 5])
    u_bc = apply_boundary_conditions(u)
    assert u_bc[0] == u_bc[-2]
    assert u_bc[-1] == u_bc[1]

def test_compute_second_derivative():
    u = np.array([0, 1, 4, 9, 16, 25])
    dx = 1.0
    u_xx = compute_second_derivative(u, dx)
    assert len(u_xx) == len(u)
    assert np.allclose(u_xx[1:-1], 2)  

def test_cal_rhs():
    u = np.array([0, 1, 2, 1, 0])
    dx = 0.1
    nu = 0.1
    rhs = cal_rhs(u, dx, nu)
    assert len(rhs) == len(u)

def test_rk2_step():
    u = np.array([0, 1, 2, 1, 0])
    dx, dt, nu = 0.1, 0.01, 0.1
    u_new = rk2_step(u, dx, dt, nu)
    assert len(u_new) == len(u)
    assert not np.array_equal(u, u_new) 

def test_run_simulation():
    nx, L, nu, dt, nt = 100, 1.0, 0.1, 0.001, 10
    x, u_initial, u_final = run_simulation(nx, L, nu, dt, nt)
    assert len(x) == nx
    assert len(u_initial) == nx
    assert len(u_final) == nx
    assert not np.array_equal(u_initial, u_final) 

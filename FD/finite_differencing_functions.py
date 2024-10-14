import numpy as np

#######################
## General functions ##
#######################

def initialize(nx, L):
    """ Initialize the domain and initial condition. """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    u_initial = np.exp(-100 * (x - 0.5 * L) ** 2)
    return x, u_initial, dx

def apply_boundary_conditions(u):
    """ Apply boundary conditions (periodic in this case). """
    u_new = np.copy(u)
    u_new[0] = u[-2]
    u_new[-1] = u[1]
    return u_new

def compute_second_derivative(u, dx):
    """ Compute the second derivative of u using a three-point stencil. """
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    u_xx = apply_boundary_conditions(u_xx)
    return u_xx


################
## Advective  ##
################

def cal_rhs_advectice(u, dx, viscosity=False):
    """ Calculate rhs using the second-order upwind scheme. """
    rhs = np.zeros_like(u)

    rhs[u>=0] = - (u * (u - np.roll(u,+1)))[u>=0]/dx
    rhs[u<0] = - (u * (np.roll(u,-1)) - u)[u<0]/dx

    return rhs

def time_steping_advective(u, dx, dt):
    """ First order integration in time """
    u_new = np.zeros_like(u)
    alpha = dt / dx
    u_new = u + dt * cal_rhs_advectice(u,dx)

    u_new = apply_boundary_conditions(u_new)
    return u_new

def run_simulation_advective(nx, L, cfl, nt):
    """ Run the advection simulation and return results. """
    x, u_initial, dx = initialize(nx, L)
    dt = cfl*dx
    tlim = dt * nt
    u = u_initial.copy()
    
    # Time-stepping loop
    for _ in range(nt):
        u = time_steping_advective(u, dx, dt)

    return x, u_initial, u, tlim



####################
## Add Viscocity  ##
####################


def cal_rhs_viscocity(u, dx, nu):
    """ Calculate rhs using the second-order upwind scheme. """
    rhs = np.zeros_like(u)
    rhs = - 0.5 * (u * (np.roll(u,-1) - np.roll(u, +1))) / dx
    
    # Add viscosity term
    rhs += nu * compute_second_derivative(u, dx)
    
    rhs = apply_boundary_conditions(rhs)
    return rhs


def rk2_step_viscocity(u, dx, dt, nu):
    """ Perform one time step using RK2 integration method. """
    # Stage 1
    k1 = dt * cal_rhs_viscocity(u, dx, nu)
    u_star = u + k1
    
    # Stage 2
    k2 = dt * cal_rhs_viscocity(u_star, dx, nu)
    u_next = u + 0.5 * (k1 + k2)
    
    u_next = apply_boundary_conditions(u_next)
    return u_next

def run_simulation_viscocity(nx, L, nu, dt, nt):
    """ Run the advection simulation and return results. """
    x, u_initial, dx = initialize(nx, L)
    u = u_initial.copy()
    
    # Time-stepping loop
    for _ in range(nt):
        u = rk2_step_viscocity(u, dx, dt, nu)
    
    return x, u_initial, u


#################################
## Keisler Oliver Dissipaiton  ##
#################################

def keisler_oliver_dissipation(u, dx):
    """ Compute Keisler-Oliver dissipation term. """
    dissipation = np.zeros_like(u)
    # First-order dissipation term
    dissipation[1:-1] += (u[2:] - u[1:-1]) / dx
    dissipation[1:-1] -= (u[1:-1] - u[:-2]) / dx
    dissipation[1:-1] *= -1
    # Second-order dissipation term
    dissipation[1:-1] += (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)
    dissipation = apply_boundary_conditions(dissipation)
    return dissipation

def cal_rhs_KOdissipation(u, dx, nu):
    """ Calculate the right-hand side with dissipation. """
    rhs = np.zeros_like(u)
    rhs[1:-1] = -u[1:-1] * (u[2:] - u[:-2]) / (2 * dx)
    rhs[1:-1] += nu * compute_second_derivative(u, dx)[1:-1]
    rhs[1:-1] += keisler_oliver_dissipation(u, dx)[1:-1]
    rhs = apply_boundary_conditions(rhs)
    return rhs

def rk2_step_KOdissipation(u, dx, dt, nu):
    """ Perform one time step using RK2 integration method. """
    # Stage 1
    k1 = dt * cal_rhs_KOdissipation(u, dx, nu)
    u_star = u + k1
    
    # Stage 2
    k2 = dt * cal_rhs_KOdissipation(u_star, dx, nu)
    u_next = u + 0.5 * (k1 + k2)
    
    u_next = apply_boundary_conditions(u_next)
    return u_next

def run_simulation_KOdissipation(nx, L, nu, dt, nt):
    """ Run the Berger's equation simulation and return results. """
    x, u_initial, dx = initialize(nx, L)
    u = u_initial.copy()
    
    # Time-stepping loop
    for _ in range(nt):
        u = rk2_step_KOdissipation(u, dx, dt, nu)
    
    return x, u_initial, u
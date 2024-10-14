import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Simulation parameters
width, height = 200, 80  # Grid size
viscosity = 0.02         # Fluid viscosity
tau = 3 * viscosity + 0.5 # Relaxation time
iterations = 1000         # Number of simulation steps

# Velocity vectors and weights
v = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], 
              [1, 1], [-1, 1], [-1, -1], [1, -1]])
w = np.array([4/9] + [1/9]*4 + [1/36]*4)

# Initialization of fields
def initialize_fields():
    rho = np.ones((height, width))  # Fluid density
    vel = np.zeros((height, width, 2))  # Velocity field
    f = np.ones((height, width, 9)) * rho[:, :, None] / 9  # Distribution functions
    return rho, vel, f

# Equilibrium distribution function
def equilibrium(rho, vel):
    u2 = vel[..., 0]**2 + vel[..., 1]**2
    feq = np.zeros((height, width, 9))
    for i in range(9):
        vu = v[i, 0] * vel[..., 0] + v[i, 1] * vel[..., 1]
        feq[..., i] = rho * w[i] * (1 + 3*vu + 9/2*vu**2 - 3/2*u2)
    return feq

# Streaming step
def streaming(f):
    f_streamed = np.zeros_like(f)
    for i in range(9):
        f_streamed[..., i] = np.roll(np.roll(f[..., i], v[i, 0], axis=1), v[i, 1], axis=0)
    return f_streamed

# Collision step
def collision(f, feq):
    return f - (f - feq) / tau

# Boundary conditions
def apply_boundary_conditions(vel):
    vel[0, :, :] = 0  # No-slip at top boundary
    vel[-1, :, :] = 0  # No-slip at bottom boundary

# Main simulation loop
def simulate():
    rho, vel, f = initialize_fields()

    for t in range(iterations):
        # Compute macroscopic quantities
        rho = np.sum(f, axis=2)
        vel = np.einsum('ijk,kl->ijl', f, v) / rho[..., None]

        # Apply boundary conditions
        apply_boundary_conditions(vel)

        # Compute equilibrium distribution
        feq = equilibrium(rho, vel)

        # Collision step
        f = collision(f, feq)

        # Streaming step
        f = streaming(f)

        # Visualization every 100 iterations
        if t % 100 == 0:
            plt.imshow(np.sqrt(vel[..., 0]**2 + vel[..., 1]**2), cmap=cm.viridis)
            plt.colorbar()
            plt.title(f'Iteration {t}')
            plt.pause(0.01)

# Run the simulation
simulate()
plt.show()


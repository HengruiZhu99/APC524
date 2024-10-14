def initialize(nx, L):
    """
    Initialize the domain and initial condition.

    Args:
        nx (int): Number of spatial grid points.
        L (float): Length of the spatial domain.
    
    Returns:
        tuple: A tuple containing the spatial grid points (array), the initial values (array), and the grid spacing (float).
    """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    u_initial = np.exp(-100 * (x - 0.5 * L) ** 2)
    return x, u_initial, dx


def test_boundary_conditions():
    u = np.array([1, 2, 3, 4, 5])
    u_new = apply_boundary_conditions(u)
    
    if u_new[0] != u[-2]:
        raise ValueError(f"Error: First boundary condition not correctly applied.")
    
    if u_new[-1] != u[1]:
        raise ValueError(f"Error: Last boundary condition not correctly applied.")

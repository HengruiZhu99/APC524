def test_initialize_function():
    nx = 100
    L = 1
    
    grid_points, initial_values, dx = initialize(nx, L)
    
    assert len(grid_points) == nx                                                                     
    assert len(initial_values) == nx                                                                    
    
    expected_dx = L / (nx - 1)
    assert dx == expected_dx   

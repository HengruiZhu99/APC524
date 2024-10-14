def test_cal_rhs_single_positive_negative_zero():
    u = np.array([2.0, -3.0, 0.0])
    dx = 0.1
    expected_rhs = np.array([-40.0, 150.0, 0.0])
    rhs = cal_rhs(u, dx)
    np.testing.assert_array_almost_equal(rhs, expected_rhs, decimal=6)

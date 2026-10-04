"""Tests for MLPG module (Gates V2-1 and V2-3)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthesis.mlpg import compute_dynamic_features, build_w_matrix, mlpg

def test_dynamic_feature_extraction():
    """Gate V2-1: Unit test delta/delta-delta convolutions and edge padding correctness."""
    print("Running Gate V2-1: Dynamic Feature Extraction Correctness...")
    T, D = 5, 2
    # Known synthetic static trajectory
    C = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [4.0, 30.0],
        [7.0, 40.0],
        [11.0, 50.0]
    ], dtype=np.float64)
    
    O = compute_dynamic_features(C)
    
    # Check shape
    assert O.shape == (T, 3*D)
    
    # Check static
    np.testing.assert_array_almost_equal(O[:, :D], C)
    
    # Hand-calculate expected deltas for dim 0
    # C[:, 0] = [1, 2, 4, 7, 11]
    # padded = [1, 1, 2, 4, 7, 11, 11]
    # delta[0] = -0.5*1 + 0.5*2 = 0.5
    # delta[1] = -0.5*1 + 0.5*4 = 1.5
    # delta[2] = -0.5*2 + 0.5*7 = 2.5
    # delta[3] = -0.5*4 + 0.5*11 = 3.5
    # delta[4] = -0.5*7 + 0.5*11 = 2.0
    expected_delta_0 = np.array([0.5, 1.5, 2.5, 3.5, 2.0])
    np.testing.assert_array_almost_equal(O[:, D], expected_delta_0)
    
    # Hand-calculate expected delta-deltas for dim 0
    # dd[0] = 1 - 2*1 + 2 = 1 (Wait, dd_0 = c_{-1} - 2c_0 + c_1 = 1 - 2(1) + 2 = 1)
    # dd[1] = 1 - 2*2 + 4 = 1
    # dd[2] = 2 - 2*4 + 7 = 1
    # dd[3] = 4 - 2*7 + 11 = 1
    # dd[4] = 7 - 2*11 + 11 = -4
    expected_dd_0 = np.array([1.0, 1.0, 1.0, 1.0, -4.0])
    np.testing.assert_array_almost_equal(O[:, 2*D], expected_dd_0)
    
    print("Gate V2-1 Passed!")

def test_mlpg_matrix_sanity():
    """Gate V2-3: MLPG Matrix Mathematical Sanity Test."""
    print("Running Gate V2-3: MLPG Matrix Mathematical Sanity Test...")
    T, D = 10, 2
    # a. Create a small known synthetic static trajectory C
    np.random.seed(42)
    C = np.random.randn(T, D).astype(np.float64)
    
    # b. Compute its delta and delta-delta features using the exact chosen boundary convention
    O_expected = compute_dynamic_features(C)
    
    # c. Construct O = W*C (to verify W matrix correctness)
    W = build_w_matrix(T)
    for d in range(D):
        # O_d via W*C
        C_d = C[:, d]
        O_d = W @ C_d # shape [3T]
        
        # O_expected_d
        O_expected_d = np.concatenate([O_expected[:, d], O_expected[:, D+d], O_expected[:, 2*D+d]])
        
        np.testing.assert_array_almost_equal(O_d, O_expected_d)
    
    print("  W matrix structure exactly matches feature extraction boundary logic.")
    
    # d. Feed the resulting synthetic observation statistics into MLPG
    # We set means = O_expected, and variances = small uniform values (to give equal weight)
    means = O_expected
    variances = np.ones_like(means) * 1.0
    
    C_hat, metrics = mlpg(means, variances)
    
    # e. Verify that the recovered C_hat matches the known C within numerical tolerance
    assert metrics["solver_success"]
    print(f"  Max residual norm: {max(metrics['residual_norms']):.2e}")
    np.testing.assert_array_almost_equal(C, C_hat, decimal=5)
    
    print("Gate V2-3 Passed!")

if __name__ == "__main__":
    test_dynamic_feature_extraction()
    test_mlpg_matrix_sanity()

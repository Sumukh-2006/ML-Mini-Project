"""Maximum Likelihood Parameter Generation (MLPG) solver for V2.

Implements trajectory generation with dynamic features (Delta + Delta-Delta)
using strict edge-padding boundaries and banded matrix solvers.
"""
import numpy as np
import scipy.linalg

def compute_dynamic_features(static_features):
    """
    Computes utterance-wide static, delta, and delta-delta features.
    
    Args:
        static_features: [T, D] numpy array
    Returns:
        O: [T, 3*D] numpy array (Static, Delta, Delta-Delta)
    """
    T, D = static_features.shape
    O = np.zeros((T, 3 * D), dtype=np.float64)
    
    # 1. Static
    O[:, :D] = static_features
    
    # 2. Delta
    # W1 = [-0.5, 0.0, 0.5]
    padded = np.pad(static_features, ((1, 1), (0, 0)), mode='edge')
    delta = -0.5 * padded[:-2] + 0.5 * padded[2:]
    O[:, D:2*D] = delta
    
    # 3. Delta-Delta
    # W2 = [1.0, -2.0, 1.0]
    delta_delta = padded[:-2] - 2.0 * static_features + padded[2:]
    O[:, 2*D:] = delta_delta
    
    return O

def build_w_matrix(T):
    """
    Constructs the banded W matrix of shape [3T, T].
    W maps a [T, 1] static sequence to a [3T, 1] observation sequence [c, delta, delta-delta]^T
    strictly applying edge-padding at boundaries.
    """
    W_0 = np.eye(T)
    
    W_1 = np.zeros((T, T))
    for t in range(T):
        if t == 0:
            W_1[t, 0] = -0.5
            if T > 1:
                W_1[t, 1] = 0.5
            else:
                W_1[t, 0] += 0.5
        elif t == T - 1:
            W_1[t, t-1] = -0.5
            W_1[t, t] = 0.5
        else:
            W_1[t, t-1] = -0.5
            W_1[t, t+1] = 0.5
            
    W_2 = np.zeros((T, T))
    for t in range(T):
        if t == 0:
            W_2[t, 0] = -1.0
            if T > 1:
                W_2[t, 1] = 1.0
            else:
                W_2[t, 0] += 1.0
        elif t == T - 1:
            W_2[t, t-1] = 1.0
            W_2[t, t] = -1.0
        else:
            W_2[t, t-1] = 1.0
            W_2[t, t] = -2.0
            W_2[t, t+1] = 1.0
            
    W = np.vstack([W_0, W_1, W_2]) # shape [3T, T]
    return W

def get_banded_form_solveh(R):
    """
    Extracts the lower triangular banded form of symmetric matrix R for solveh_banded.
    R is pentadiagonal, so lower bandwidth is 2.
    """
    T = R.shape[0]
    ab = np.zeros((3, T))
    ab[0, :] = np.diag(R)
    ab[1, :-1] = np.diag(R, k=-1)
    ab[2, :-2] = np.diag(R, k=-2)
    return ab

def mlpg(means, variances):
    """
    Solves the MLPG optimization problem: (W^T U^{-1} W) C = W^T U^{-1} mu.
    
    Args:
        means: [T, 3*D] numpy array (means for static, delta, delta-delta)
        variances: [T, 3*D] numpy array (variances for static, delta, delta-delta)
    Returns:
        C: [T, D] optimized static trajectory
        metrics: dict of solver metrics (for numerical stability diagnostics)
    """
    T, full_D = means.shape
    D = full_D // 3
    
    W = build_w_matrix(T) # [3T, T]
    
    C = np.zeros((T, D), dtype=np.float32)
    
    metrics = {
        "solver_success": True,
        "residual_norms": [],
        "min_precision": float("inf"),
        "max_precision": float("-inf")
    }
    
    # Precision matrix diagonal
    U_inv = 1.0 / variances # [T, 3*D]
    
    metrics["min_precision"] = float(np.min(U_inv))
    metrics["max_precision"] = float(np.max(U_inv))
    
    for d in range(D):
        # Extract means and precisions for dimension d
        mu_d = np.concatenate([means[:, d], means[:, D+d], means[:, 2*D+d]]) # [3T]
        U_inv_d = np.concatenate([U_inv[:, d], U_inv[:, D+d], U_inv[:, 2*D+d]]) # [3T]
        
        # Construct R = W^T U_inv W and r = W^T U_inv mu
        W_T_U_inv_d = W.T * U_inv_d[None, :] # shape [T, 3T]
        R = W_T_U_inv_d @ W # shape [T, T]
        r = W_T_U_inv_d @ mu_d # shape [T]
        
        ab = get_banded_form_solveh(R)
        
        try:
            # Solve R C_d = r using Cholesky banded solver
            c_d = scipy.linalg.solveh_banded(ab, r, lower=True)
            C[:, d] = c_d
            
            # Check numerical stability (residual)
            res = R @ c_d - r
            metrics["residual_norms"].append(float(np.linalg.norm(res)))
            
            # Check for NaN or Inf
            if not np.all(np.isfinite(c_d)):
                metrics["solver_success"] = False
                
        except Exception as e:
            metrics["solver_success"] = False
            # Fallback to standard solver if banded fails (e.g. ill-conditioned)
            try:
                c_d = np.linalg.solve(R, r)
                C[:, d] = c_d
            except:
                pass
                
    return C, metrics

import torch
import numpy as np
from scipy.linalg import eigh, sqrtm
import matplotlib.pyplot as plt

def generate_spd_from_loguniform(d, log_range=(-np.log(2), np.log(2))):
    """Generate SPD matrix with log-uniform eigenvalues and random orthogonal eigenvectors."""
    # Sample log-eigenvalues uniformly
    log_eigs = np.random.uniform(log_range[0], log_range[1], d)
    eigs = np.exp(log_eigs)

    # Generate a random orthogonal matrix (eigenvectors)
    Q, _ = np.linalg.qr(np.random.randn(d, d))  # QR factorization gives orthogonal Q

    # Construct SPD matrix: Σ = Q Λ Q^T
    Sigma = Q @ np.diag(eigs) @ Q.T
    return Sigma

def compute_ot_map_matrix(Sigma0, Sigma1):
    """Compute the optimal transport linear map matrix Gamma from Σ0 to Σ1."""
    # sqrt_Sigma0 = np.linalg.cholesky(Sigma0)
    # inv_sqrt_Sigma0 = np.linalg.inv(sqrt_Sigma0)
    # middle = sqrt_Sigma0 @ Sigma1 @ sqrt_Sigma0.T
    # sqrt_middle = np.linalg.cholesky(middle)
    # Gamma = inv_sqrt_Sigma0.T @ sqrt_middle @ inv_sqrt_Sigma0
    
    sqrt_Sigma0 = sqrtm(Sigma0)
    inv_sqrt_Sigma0 = np.linalg.inv(sqrt_Sigma0)
    middle = sqrt_Sigma0 @ Sigma1 @ sqrt_Sigma0
    sqrt_middle = sqrtm(middle)
    Gamma = inv_sqrt_Sigma0 @ sqrt_middle @ inv_sqrt_Sigma0
    return Gamma.real # Discard any imaginary parts due to numerical error

def compute_ot_map_matrix_highdim(Sigma0, Sigma1, eps=1e-12):
    """
    Compute the optimal transport linear map Gamma from Σ0 to Σ1.
    Works for high-dimensional (e.g., 784-dim) covariances using eigendecomposition.
    """
    # Eigendecomposition of Sigma0
    eigvals0, eigvecs0 = eigh(Sigma0)
    eigvals0 = np.clip(eigvals0, eps, None)  # avoid tiny/negative eigenvalues

    # Sigma0^(1/2) and Sigma0^(-1/2)
    sqrt_Sigma0 = eigvecs0 @ np.diag(np.sqrt(eigvals0)) @ eigvecs0.T
    inv_sqrt_Sigma0 = eigvecs0 @ np.diag(1/np.sqrt(eigvals0)) @ eigvecs0.T

    # Middle term and its sqrt
    middle = sqrt_Sigma0 @ Sigma1 @ sqrt_Sigma0
    eigvals_mid, eigvecs_mid = eigh(middle)
    eigvals_mid = np.clip(eigvals_mid, 0, None)
    sqrt_middle = eigvecs_mid @ np.diag(np.sqrt(eigvals_mid)) @ eigvecs_mid.T

    # OT map
    Gamma = inv_sqrt_Sigma0 @ sqrt_middle @ inv_sqrt_Sigma0
    return Gamma.real

def wasserstein_2_squared(Sigma0, Sigma1, m0=0, m1=0):
    """
    Compute the Wasserstein-2 squared distance(= True transport cost) between two multivariate Gaussians.
    
    Parameters:
    m0 (np.ndarray): Mean of the first Gaussian distribution (d-dimensional vector)
    m1 (np.ndarray): Mean of the second Gaussian distribution (d-dimensional vector)
    Sigma0 (np.ndarray): Covariance of the first Gaussian distribution (d x d matrix)
    Sigma1 (np.ndarray): Covariance of the second Gaussian distribution (d x d matrix)
    
    Returns:
    float: Wasserstein-2 squared distance between the two distributions
    """
    # Compute the first term: ||m0 - m1||^2
    mean_term = np.linalg.norm(m0 - m1)**2

    # Compute the second term: Tr(Sigma0 + Sigma1 - 2(Sigma0^(1/2) Sigma1 Sigma0^(1/2))^(1/2))
    sqrt_Sigma0 = sqrtm(Sigma0) # Sigma0^(1/2)
    middle_term = np.dot(sqrt_Sigma0, np.dot(Sigma1, sqrt_Sigma0))
    middle_term_sqrt = sqrtm(middle_term)  # Square root of matrix
    
    # Compute the trace term
    trace_term = np.trace(Sigma0 + Sigma1 - 2 * middle_term_sqrt)
    
    # The Wasserstein-2 squared distance
    W2_squared = mean_term + trace_term
    
    return W2_squared

def wasserstein_2_squared_highdim(Sigma0, Sigma1, m0=None, m1=None, eps=1e-12):
    """Compute W2^2 between high-dimensional Gaussians."""
    if m0 is None: m0 = np.zeros(Sigma0.shape[0])
    if m1 is None: m1 = np.zeros(Sigma1.shape[0])

    mean_term = np.linalg.norm(m0 - m1)**2

    # Eigendecomposition-based sqrt
    eigvals0, eigvecs0 = eigh(Sigma0)
    eigvals0 = np.clip(eigvals0, eps, None)
    sqrt_Sigma0 = eigvecs0 @ np.diag(np.sqrt(eigvals0)) @ eigvecs0.T

    middle = sqrt_Sigma0 @ Sigma1 @ sqrt_Sigma0
    eigvals_mid, _ = eigh(middle)
    eigvals_mid = np.clip(eigvals_mid, 0, None)
    trace_term = np.trace(Sigma0 + Sigma1 - 2*np.diag(np.sqrt(eigvals_mid)))

    return mean_term + trace_term

# ---------------------------------------------------------------------------
# UVP
# ---------------------------------------------------------------------------

def compute_uvp(
    x_batch: torch.Tensor,          # source samples  (B, d)
    z_batch: torch.Tensor,          # target samples  (B, d)
    y_star_batch: torch.Tensor,     # fixed-point solution y* (B, d)
    model,                          # ImplicitICNN with grad_g method
    Gamma: np.ndarray,
    Gamma_inv: np.ndarray,
    var0: float,
    var1: float,
    dim: int,
):
    """Return (forward_UVP, backward_UVP) as percentages."""
    T_X_true  = x_batch.detach().cpu().numpy() @ Gamma.T
    T_Z_true  = z_batch.detach().cpu().numpy() @ Gamma_inv.T

    xhat = z_batch - model.grad_g(y_star_batch)   # ≡ y*
    zhat = x_batch + model.grad_g(x_batch)

    forward_UVP  = dim * 100 * ((T_X_true - zhat.detach().cpu().numpy()) ** 2).mean() / var1
    backward_UVP = dim * 100 * ((T_Z_true - xhat.detach().cpu().numpy()) ** 2).mean() / var0
    return forward_UVP, backward_UVP

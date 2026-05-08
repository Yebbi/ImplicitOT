"""
utils/metrics.py — OT map estimation and UVP error for Gaussian benchmarks.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.linalg import sqrtm


# ---------------------------------------------------------------------------
# Ground-truth OT map for Gaussians
# ---------------------------------------------------------------------------

def compute_ot_map_matrix(
    Sigma0: np.ndarray,
    Sigma1: np.ndarray,
) -> np.ndarray:
    """Compute the linear OT map T = Sigma0^{-1/2} (Sigma0^{1/2} Sigma1 Sigma0^{1/2})^{1/2} Sigma0^{-1/2}.

    Returns Gamma such that T(x) = Gamma @ x maps N(0, Sigma0) → N(0, Sigma1).
    """
    S0_sqrt = sqrtm(Sigma0).real
    S0_inv_sqrt = np.linalg.inv(S0_sqrt)
    M = S0_sqrt @ Sigma1 @ S0_sqrt
    Gamma = S0_inv_sqrt @ sqrtm(M).real @ S0_inv_sqrt
    return Gamma.real


def wasserstein_2_squared(
    Sigma0: np.ndarray,
    Sigma1: np.ndarray,
) -> float:
    """Closed-form W2² between N(0,Σ₀) and N(0,Σ₁)."""
    S0_sqrt = sqrtm(Sigma0).real
    M = sqrtm(S0_sqrt @ Sigma1 @ S0_sqrt).real
    return float(np.trace(Sigma0) + np.trace(Sigma1) - 2 * np.trace(M))


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
) -> Tuple[float, float]:
    """Return (forward_UVP, backward_UVP) as percentages."""
    T_X_true  = x_batch.detach().cpu().numpy() @ Gamma.T
    T_Z_true  = z_batch.detach().cpu().numpy() @ Gamma_inv.T

    xhat = z_batch - model.grad_g(y_star_batch)   # ≡ y*
    zhat = x_batch + model.grad_g(x_batch)

    forward_UVP  = dim * 100 * ((T_X_true - zhat.detach().cpu().numpy()) ** 2).mean() / var1
    backward_UVP = dim * 100 * ((T_Z_true - xhat.detach().cpu().numpy()) ** 2).mean() / var0
    return forward_UVP, backward_UVP

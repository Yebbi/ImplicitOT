"""
utils/losses.py — MMD-RBF kernel loss and OT dual objective.
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Unbiased MMD² with multi-scale RBF kernel and median-heuristic bandwidths.

    Args:
        x: Tensor of shape (n, d).
        y: Tensor of shape (m, d).
        bandwidths: Optional pre-computed bandwidths.  If None, estimated
                    from the median pairwise squared distance of x vs y.

    Returns:
        Scalar MMD² estimate.
    """
    x = x.reshape(x.size(0), -1)
    y = y.reshape(y.size(0), -1)

    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)

    if bandwidths is None:
        with torch.no_grad():
            median_sq = torch.median(xy)
            bandwidths = [median_sq * f for f in [0.25, 0.5, 1.0, 2.0, 4.0]]

    K_xx = sum(torch.exp(-xx / (2 * bw)) for bw in bandwidths)
    K_yy = sum(torch.exp(-yy / (2 * bw)) for bw in bandwidths)
    K_xy = sum(torch.exp(-xy / (2 * bw)) for bw in bandwidths)

    n, m = x.size(0), y.size(0)
    return (
        (K_xx.sum() - K_xx.diagonal().sum()) / (n * (n - 1))
        - 2 * K_xy.sum() / (n * m)
        + (K_yy.sum() - K_yy.diagonal().sum()) / (m * (m - 1))
    )


# ---------------------------------------------------------------------------
# OT dual loss
# ---------------------------------------------------------------------------

def ot_dual_loss(
    g_source: torch.Tensor,
    g_ystar: torch.Tensor,
) -> torch.Tensor:
    """Class-conditional OT dual objective.

    L_CC = E_mu[g(x, k)] - E_nu[g(y*, k)]

    The caller is responsible for computing g_source = g(x, k) and
    g_ystar = g(y*, k) where y* is the implicit fixed-point solution.

    Args:
        g_source: (B, 1) network output evaluated on source latents.
        g_ystar:  (B, 1) network output evaluated on fixed-point solution y*.

    Returns:
        Scalar loss (to be minimised).
    """
    return g_source.mean() - g_ystar.mean()


def vae_loss(x, x_hat, mu, logvar, beta: float):
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon + beta * kl, recon, kl
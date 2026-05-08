"""
utils/losses.py — OT dual objective for ICNN-based transport.
"""

import torch


def ot_dual_loss(
    g_source: torch.Tensor,
    g_ystar: torch.Tensor,
) -> torch.Tensor:
    """Semi-dual OT loss.

    L = E_mu[g(x)] - E_nu[0.5*||z - y*||² + g(y*)]

    Args:
        g_source: scalar — E_mu[g(x)], already averaged over the batch.
        g_ystar:  scalar — E_nu[0.5*||z-y*||² + g(y*)], already averaged.

    Returns:
        Scalar loss to be minimised.
    """
    return g_source - g_ystar

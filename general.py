"""
utils/general.py — Filesystem helpers and structured logging.
"""

from __future__ import annotations
import os
import random
import numpy as np
import torch


def mkdir_ifnotexists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


def append_log(path: str, msg: str) -> None:
    """Append *msg* to *path*, creating the file if necessary."""
    with open(path, "a") as f:
        f.write(msg)

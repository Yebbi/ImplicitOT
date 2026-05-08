from __future__ import annotations
import torch
import numpy as np
from configs.config_gaussians import DataConfig

class GaussianDataset:
    """Loads source (mu) and target (nu) Gaussian distributions from .npz.

    Holds the full pre-sampled arrays in memory and provides random
    mini-batch draws without replacement.
    """

    def __init__(self, cfg: DataConfig, dtype: torch.dtype = torch.float32) -> None:
        raw = np.load(cfg.npz_path())
        self.Sigma0: np.ndarray = raw["Sigma0"]   # source covariance
        self.Sigma1: np.ndarray = raw["Sigma1"]   # target covariance
        self.dim: int = cfg.dim
        self.dtype = dtype

        rng = np.random.default_rng()
        self.data_source = rng.multivariate_normal(
            mean=np.zeros(cfg.dim), cov=self.Sigma0, size=cfg.n_samples
        )
        self.data_target = rng.multivariate_normal(
            mean=np.zeros(cfg.dim), cov=self.Sigma1, size=cfg.n_samples
        )

        self.var0: float = float(np.trace(self.Sigma0))
        self.var1: float = float(np.trace(self.Sigma1))

    def sample(
        self,
        batch_size: int,
        device: str,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw *batch_size* samples from source and target without replacement."""
        idx_src = np.random.choice(self.data_source.shape[0], batch_size, replace=False)
        idx_tgt = np.random.choice(self.data_target.shape[0], batch_size, replace=False)

        x = torch.tensor(
            self.data_source[idx_src], dtype=self.dtype, device=device,
            requires_grad=requires_grad,
        )
        z = torch.tensor(
            self.data_target[idx_tgt], dtype=self.dtype, device=device,
            requires_grad=requires_grad,
        )
        return x, z

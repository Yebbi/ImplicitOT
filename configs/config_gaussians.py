"""
configs/config.py — All hyperparameters for Gaussian OT experiments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    dim: int = 32
    n_samples: int = 100_000
    data_dir: str = "./codes/data/Gaussians"

    def npz_path(self) -> str:
        return os.path.join(self.data_dir, f"{self.dim}D_Test_Distributions.npz")


@dataclass
class ModelConfig:
    activation: str = "softplus"
    rank: int = 1
    strong_convexity: float = 1e-4

    # hidden layer sizes are auto-computed from dim if not set explicitly
    hidden_layer_sizes: Optional[List[int]] = None

    def resolve_hidden(self, dim: int) -> List[int]:
        if self.hidden_layer_sizes is not None:
            return self.hidden_layer_sizes
        return [max(2 * dim, 64), max(2 * dim, 64), max(dim, 32)]


@dataclass
class TrainConfig:
    max_epochs: int = 5_000
    batch_size: int = 2_048
    sampling_freq: int = 3        # resample data every N epochs
    lr: float = 1e-3              # overridden for dim >= 16 unless explicit
    lr_auto_scale: bool = True    # set lr=1e-4 when dim >= 16
    grad_clip: float = 10.0
    log_freq: int = 100
    save_freq: int = 100


@dataclass
class SolverConfig:
    tol: float = 1e-3
    max_iters: int = 1_000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    # runtime
    device: str = "cuda:0"
    output_dir: str = "./exps/Accuracy_gaussians"
    log_filename: str = "logs_icnn_convexity_softplus"
    seed: int = 42

    def exp_dir(self) -> str:
        return os.path.join(self.output_dir, f"{self.data.dim}D")

    def log_path(self) -> str:
        return os.path.join(self.exp_dir(), f"{self.log_filename}.txt")

    def resolve(self) -> None:
        """Apply dimension-dependent defaults after construction."""
        if self.train.lr_auto_scale and self.data.dim >= 16:
            self.train.lr = 1e-4

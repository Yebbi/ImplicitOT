"""
configs/config.py — All hyperparameters and experiment settings.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class VAEConfig:
    latent_dim: int = 15
    num_classes: int = 10
    beta_max_source: float = 1.0    # beta used when training source VAE (fmnist)
    beta_max_target: float = 0.1    # beta used when training target VAE (mnist)


@dataclass
class DataConfig:
    source_dataset: str = "fmnist"   # "fmnist" | "mnist"
    target_dataset: str = "mnist"    # "fmnist" | "mnist"
    img_size: int = 32
    batch_size: int = 64
    num_workers: int = 2


@dataclass
class ModelConfig:
    hidden_structure: Tuple[int, ...] = (512, 512, 512, 512)
    activation: str = "softplus"


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    max_iters: int = 10_000         # inner fixed-point iterations
    tol: float = 1e-3               # inner solver tolerance
    lambda_mmd: float = 1e-3        # MMD regularisation weight (0 = disabled)
    mmd_subsample: int = 5_000      # subsample size for MMD computation
    print_freq: int = 1             # print every N epochs
    plot_freq: int = 1              # save visualisation every N epochs
    inner_grad_norm_threshold: float = 10.0   # early-stop if exceeded


@dataclass
class EvalConfig:
    n_samples: int = 10_000
    fid_dims: int = 2048
    fid_batch_size: int = 64


@dataclass
class Config:
    # sub-configs
    vae: VAEConfig = field(default_factory=VAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # paths
    source_vae_path: Optional[str] = None   # auto-resolved if None
    target_vae_path: Optional[str] = None   # auto-resolved if None
    save_path: Optional[str] = None         # auto-resolved if None
    save_fig_path: str = "implicit_ot_latent_progress.png"
    output_dir: str = "outputs"

    # runtime
    device: str = "cuda:0"
    precision: str = "double"       # "single" | "double"
    seed: int = 42

    def resolve_paths(self) -> None:
        """Fill in auto-generated paths when not explicitly provided."""
        v = self.vae
        t = self.train
        if self.source_vae_path is None:
            self.source_vae_path = os.path.join(
                ".vae/ckpts",
                f"cvae_{self.data.source_dataset}"
                f"_beta{v.beta_max_source}"
                f"_latent{v.latent_dim}_bce.pth",
            )
        if self.target_vae_path is None:
            self.target_vae_path = os.path.join(
                ".vae/ckpts",
                f"cvae_{self.data.target_dataset}"
                f"_beta{v.beta_max_target}"
                f"_latent{v.latent_dim}_bce.pth",
            )
        if self.save_path is None:
            self.save_path = os.path.join(
                self.output_dir,
                f"implicit_ot_latent_mmd{t.lambda_mmd}.pth",
            )

    def validate_vae_paths(self) -> None:
        """Raise FileNotFoundError with a clear message if a VAE ckpt is missing."""
        for label, path in [
            ("source VAE", self.source_vae_path),
            ("target VAE", self.target_vae_path),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{label} checkpoint not found: '{path}'\n"
                    f"  Pass the correct path via --source_vae_path / --target_vae_path,\n"
                    f"  or make sure the file exists at the auto-resolved location."
                )

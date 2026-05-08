"""
main.py — Entry point for Gaussian OT training.

Usage
-----
# Default (dim=32):
python main.py

# Custom:
python main.py --dim 64 --tol 1e-4 --max_iter 2000 --gpu 1 --epochs 3000
"""

from __future__ import annotations
import argparse

import torch

from configs import Config, DataConfig, ModelConfig, TrainConfig, SolverConfig
from utils.dataset import GaussianDataset
from utils.general import set_seed
from trainer import GaussianOTTrainer

from models.ImplicitICNN import ImplicitICNN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaussian OT with ImplicitICNN")

    # data
    p.add_argument("--dim",      type=int,   default=32,
                   help="Input/output dimension")
    p.add_argument("--n_samples",type=int,   default=100_000,
                   help="Number of pre-sampled points per distribution")
    p.add_argument("--data_dir", default="/home/yesom/Codes/Implicit_JF_OT/codes/datasets/Gaussians",
                   help="Directory containing {dim}D_Test_Distributions.npz")

    # model
    p.add_argument("--activation",       default="softplus")
    p.add_argument("--rank",             type=int,   default=1)
    p.add_argument("--strong_convexity", type=float, default=1e-4)
    p.add_argument("--hidden",           type=int,   nargs="+", default=None,
                   help="Hidden layer sizes (auto-computed from dim if omitted)")

    # training
    p.add_argument("--epochs",       type=int,   default=5_000)
    p.add_argument("--batch_size",   type=int,   default=2_048)
    p.add_argument("--lr",           type=float, default=None,
                   help="Learning rate (auto: 1e-3 for dim<16, 1e-4 for dim>=16)")
    p.add_argument("--sampling_freq",type=int,   default=3)
    p.add_argument("--grad_clip",    type=float, default=10.0)
    p.add_argument("--log_freq",     type=int,   default=100)
    p.add_argument("--save_freq",    type=int,   default=100)

    # solver
    p.add_argument("--tol",      type=float, default=1e-3,
                   help="Fixed-point iteration tolerance")
    p.add_argument("--max_iter", type=int,   default=1_000,
                   help="Max fixed-point iterations")

    # runtime
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output_dir", default="./exps/Test_accuracy_gaussians")
    p.add_argument("--log_name",   default="logs_icnn_convexity_softplus")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    train_cfg = TrainConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        sampling_freq=args.sampling_freq,
        grad_clip=args.grad_clip,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        lr_auto_scale=args.lr is None,
        lr=args.lr if args.lr is not None else 1e-3,
    )
    return Config(
        data=DataConfig(
            dim=args.dim,
            n_samples=args.n_samples,
            data_dir=args.data_dir,
        ),
        model=ModelConfig(
            activation=args.activation,
            rank=args.rank,
            strong_convexity=args.strong_convexity,
            hidden_layer_sizes=args.hidden,
        ),
        train=train_cfg,
        solver=SolverConfig(
            tol=args.tol,
            max_iters=args.max_iter,
        ),
        device=f"cuda:{args.gpu}",
        output_dir=args.output_dir,
        log_filename=args.log_name,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)
    cfg.resolve()

    set_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)

    print(f"Device : {cfg.device}")
    print(f"Dim    : {cfg.data.dim}")
    print(f"Data   : {cfg.data.npz_path()}")

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = GaussianDataset(cfg.data)

    # ── Model ────────────────────────────────────────────────────────────
    hidden = cfg.model.resolve_hidden(cfg.data.dim)
    model = ImplicitICNN(
        dim=cfg.data.dim,
        activation=cfg.model.activation,
        rank=cfg.model.rank,
        hidden_layer_sizes=hidden,
        strong_convexity=cfg.model.strong_convexity,
    ).to(cfg.device, dtype=torch.float64)

    print(f"Model  : ImplicitICNN | hidden={hidden} | "
          f"params={sum(p.numel() for p in model.parameters())}")

    # ── Train ────────────────────────────────────────────────────────────
    trainer = GaussianOTTrainer(cfg=cfg, model=model, dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    main()

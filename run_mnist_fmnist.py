from __future__ import annotations
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from configs.config_mnist_fmnist import Config, VAEConfig, DataConfig, ModelConfig, TrainConfig, EvalConfig
from datasets.class_ot import build_loader
from utils.evaluation import evaluate_fid, save_transport_figure
from trainers.trainer_fmnist_mnist import LatentOTTrainer

# local model imports — adjust paths to match your project layout
from models.ImplicitNet import ImplicitNet
from models.cvae import ConditionalConvVAE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class-conditional latent OT training")

    # data
    p.add_argument("--source",      default="fmnist",  help="Source dataset (fmnist | mnist)")
    p.add_argument("--target",      default="mnist",   help="Target dataset (fmnist | mnist)")
    p.add_argument("--img_size",    type=int, default=32)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)

    # VAE checkpoints
    p.add_argument("--latent_dim",        type=int,   default=15)
    p.add_argument("--num_classes",       type=int,   default=10)
    p.add_argument("--beta_source",       type=float, default=1.0,
                   help="Beta used when the source VAE was trained "
                        "(used for auto-naming the ckpt file)")
    p.add_argument("--beta_target",       type=float, default=0.1,
                   help="Beta used when the target VAE was trained")
    p.add_argument("--source_vae_path",   default=None,
                   help="Explicit path to source VAE checkpoint (.pth). "
                        "If omitted, resolved as: "
                        "cvae_{source}_beta{beta_source}_latent{latent_dim}_bce.pth")
    p.add_argument("--target_vae_path",   default=None,
                   help="Explicit path to target VAE checkpoint (.pth). "
                        "If omitted, resolved as: "
                        "cvae_{target}_beta{beta_target}_latent{latent_dim}_bce.pth")

    # model
    p.add_argument("--hidden",      type=int,   nargs="+", default=[512, 512, 512, 512])
    p.add_argument("--activation",  default="softplus")

    # training
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--max_iters",    type=int,   default=10_000)
    p.add_argument("--tol",          type=float, default=1e-3)
    p.add_argument("--lambda_mmd",   type=float, default=1e-3)
    p.add_argument("--mmd_subsample",type=int,   default=5_000)
    p.add_argument("--print_freq",   type=int,   default=1)
    p.add_argument("--plot_freq",    type=int,   default=1)

    # evaluation
    p.add_argument("--n_samples",    type=int,   default=10_000)
    p.add_argument("--skip_pre_fid", action="store_true",
                   help="Skip FID computation before training")

    # runtime
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--precision", default="double",  choices=["single", "double"])
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--save_path", default=None,
                   help="Path for best model checkpoint (auto-named if not set)")
    p.add_argument("--output_dir", default="outputs")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        vae=VAEConfig(
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            beta_max_source=args.beta_source,
            beta_max_target=args.beta_target,
        ),
        data=DataConfig(
            source_dataset=args.source,
            target_dataset=args.target,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        model=ModelConfig(
            hidden_structure=tuple(args.hidden),
            activation=args.activation,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            max_iters=args.max_iters,
            tol=args.tol,
            lambda_mmd=args.lambda_mmd,
            mmd_subsample=args.mmd_subsample,
            print_freq=args.print_freq,
            plot_freq=args.plot_freq,
        ),
        eval=EvalConfig(n_samples=args.n_samples),
        source_vae_path=args.source_vae_path,
        target_vae_path=args.target_vae_path,
        save_path=args.save_path,
        save_fig_path=os.path.join(args.output_dir, "progress.png"),
        output_dir=args.output_dir,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
    )
    cfg.resolve_paths()
    return cfg


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vae(path: str, latent_dim: int, num_classes: int,
             device: str, dtype: torch.dtype) -> ConditionalConvVAE:
    vae = ConditionalConvVAE(latent_dim=latent_dim, num_classes=num_classes)
    vae.load_state_dict(torch.load(path, map_location=device))
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # validate VAE checkpoint paths before doing anything else
    cfg.validate_vae_paths()

    # ── reproducibility ──────────────────────────────────────────────────
    set_seed(cfg.seed)

    # ── dtype ────────────────────────────────────────────────────────────
    dtype = torch.float64 if cfg.precision == "double" else torch.float32
    print(f"Device: {cfg.device}  |  Precision: {cfg.precision} ({dtype})")
    print(f"\nVAE checkpoints:")
    print(f"  source ({cfg.data.source_dataset}): {cfg.source_vae_path}")
    print(f"  target ({cfg.data.target_dataset}): {cfg.target_vae_path}")

    # ── VAEs ─────────────────────────────────────────────────────────────
    cvae_source = load_vae(cfg.source_vae_path, cfg.vae.latent_dim,
                           cfg.vae.num_classes, cfg.device, dtype)
    print(f"  source VAE loaded  (latent_dim={cvae_source.latent_dim})")

    cvae_target = load_vae(cfg.target_vae_path, cfg.vae.latent_dim,
                           cfg.vae.num_classes, cfg.device, dtype)
    print(f"  target VAE loaded  (latent_dim={cvae_target.latent_dim})")

    # ── ImplicitNet ───────────────────────────────────────────────────────
    implicit_net = ImplicitNet(
        input_dim=cfg.vae.latent_dim,
        num_classes=cfg.vae.num_classes,
        hidden_structure=cfg.model.hidden_structure,
        activation=cfg.model.activation,
    ).to(device=cfg.device, dtype=dtype)

    # load checkpoint if available
    if os.path.exists(cfg.save_path):
        implicit_net.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
        print(f"Loaded checkpoint from {cfg.save_path}")
    else:
        print(f"No checkpoint found at {cfg.save_path} — training from scratch")

    # ── Data ──────────────────────────────────────────────────────────────
    paired_loader = build_loader(cfg.data)
    print(f"\nLoader: {len(paired_loader)} batches "
          f"(batch_size={cfg.data.batch_size})")

    # ── Pre-training FID ──────────────────────────────────────────────────
    if not args.skip_pre_fid:
        fid_pre, fid_vae_pre = evaluate_fid(
            implicit_net=implicit_net,
            cvae_source=cvae_source,
            cvae_target=cvae_target,
            paired_loader=paired_loader,
            cfg=cfg.eval,
            num_classes=cfg.vae.num_classes,
            device=cfg.device,
            dtype=dtype,
            label="[PRE-TRAINING] ",
        )

    # ── Training ──────────────────────────────────────────────────────────
    trainer = LatentOTTrainer(
        cfg=cfg,
        implicit_net=implicit_net,
        cvae_source=cvae_source,
        cvae_target=cvae_target,
        paired_loader=paired_loader,
        dtype=dtype,
    )
    trainer.train()

    # ── Load best checkpoint for evaluation ───────────────────────────────
    implicit_net.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))

    # ── Final visualisation ───────────────────────────────────────────────
    x_src, x_tgt, labels = next(iter(paired_loader))
    x_src   = x_src[:16].to(device=cfg.device, dtype=dtype)
    x_tgt   = x_tgt[:16].to(device=cfg.device, dtype=dtype)
    labels  = labels[:16].to(cfg.device)

    save_transport_figure(
        implicit_net=implicit_net,
        cvae_source=cvae_source,
        cvae_target=cvae_target,
        x_src=x_src,
        x_tgt=x_tgt,
        labels=labels,
        num_classes=cfg.vae.num_classes,
        dtype=dtype,
        epoch=cfg.train.epochs,
        loss=trainer.best_loss,
        save_path=os.path.join(cfg.output_dir, f"final_mmd{cfg.train.lambda_mmd}.png"),
        n_show=16,
    )

    # ── Post-training FID ─────────────────────────────────────────────────
    fid_post, fid_vae_post = evaluate_fid(
        implicit_net=implicit_net,
        cvae_source=cvae_source,
        cvae_target=cvae_target,
        paired_loader=paired_loader,
        cfg=cfg.eval,
        num_classes=cfg.vae.num_classes,
        device=cfg.device,
        dtype=dtype,
        label="[POST-TRAINING] ",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FID COMPARISON: PRE vs POST TRAINING")
    print("=" * 70)
    if not args.skip_pre_fid:
        print(f"  OT Transported  — Pre: {fid_pre:.2f}  |  Post: {fid_post:.2f}")
        print(f"  VAE Generation  — Pre: {fid_vae_pre:.2f}  |  Post: {fid_vae_post:.2f}")
    else:
        print(f"  OT Transported  — Post: {fid_post:.2f}")
        print(f"  VAE Generation  — Post: {fid_vae_post:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

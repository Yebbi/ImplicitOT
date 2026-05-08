"""
utils/evaluation.py — FID computation and transport visualisation.
"""

from __future__ import annotations
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs import EvalConfig


# ---------------------------------------------------------------------------
# FID helpers
# ---------------------------------------------------------------------------

def _get_activations(
    images: torch.Tensor,
    model,
    device: str,
    batch_size: int,
    dims: int,
) -> np.ndarray:
    n = images.shape[0]
    pred_arr = np.empty((n, dims))

    for i in range(0, n, batch_size):
        batch = images[i : i + batch_size]
        if batch.shape[1] == 1:                        # grayscale → RGB
            batch = batch.repeat(1, 3, 1, 1)
        batch = batch.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))

        pred_arr[i : i + batch.shape[0]] = (
            pred.cpu().numpy().reshape(batch.shape[0], -1)
        )
    return pred_arr


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    cfg: EvalConfig,
    device: str,
) -> float:
    """Compute FID between two image tensors (values in [0, 1])."""
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[cfg.fid_dims]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()

    act_real = _get_activations(real_images, inception, device, cfg.fid_batch_size, cfg.fid_dims)
    act_gen  = _get_activations(generated_images, inception, device, cfg.fid_batch_size, cfg.fid_dims)

    mu_r, sigma_r = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)
    mu_g, sigma_g = np.mean(act_gen,  axis=0), np.cov(act_gen,  rowvar=False)

    return calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# ---------------------------------------------------------------------------
# Full evaluation pass
# ---------------------------------------------------------------------------

def evaluate_fid(
    implicit_net,
    cvae_source,
    cvae_target,
    paired_loader: DataLoader,
    cfg: EvalConfig,
    num_classes: int,
    device: str,
    dtype: torch.dtype,
    label: str = "",
) -> Tuple[float, float]:
    """Collect transported / real / VAE-generated images and compute FID scores."""
    print("=" * 70)
    print(f"COMPUTING FID SCORES {label}(pytorch-fid)")
    print("=" * 70)

    implicit_net.eval()
    transported_imgs, real_imgs = [], []

    with torch.no_grad():
        for x_src, x_tgt, labels in paired_loader:
            collected = sum(t.shape[0] for t in real_imgs)
            if collected >= cfg.n_samples:
                break

            x_src   = x_src.to(device=device, dtype=dtype)
            x_tgt   = x_tgt.to(device=device, dtype=dtype)
            labels  = labels.to(device)
            c_onehot = F.one_hot(labels, num_classes).to(dtype)

            x_latent, _    = cvae_source.encode(x_src, labels)
            z_transported  = x_latent + implicit_net.grad_g(x_latent, c_onehot)
            x_transported  = cvae_target.decode(z_transported, labels)

            transported_imgs.append(x_transported.cpu())
            real_imgs.append(x_tgt.cpu())

    transported_imgs = torch.cat(transported_imgs)[:cfg.n_samples]
    real_imgs        = torch.cat(real_imgs)[:cfg.n_samples]

    # VAE samples
    vae_imgs = []
    gen_per_batch = 256
    with torch.no_grad():
        for _ in range((cfg.n_samples + gen_per_batch - 1) // gen_per_batch):
            z_rand  = torch.randn(gen_per_batch, cvae_target.latent_dim, device=device, dtype=dtype)
            lbl_rand = torch.randint(0, num_classes, (gen_per_batch,)).to(device)
            vae_imgs.append(cvae_target.decode(z_rand, lbl_rand).cpu())
    vae_imgs = torch.cat(vae_imgs)[:cfg.n_samples]

    print(f"Real:         {real_imgs.shape}")
    print(f"Transported:  {transported_imgs.shape}")
    print(f"VAE samples:  {vae_imgs.shape}")

    print("\n1. FID: OT Transported vs Real target ...")
    fid_transport = compute_fid(real_imgs, transported_imgs, cfg, device)

    print("2. FID: VAE Generated vs Real target ...")
    fid_vae = compute_fid(real_imgs, vae_imgs, cfg, device)

    print("\n" + "=" * 70)
    print(f"FID RESULTS {label}")
    print("=" * 70)
    print(f"  OT Transported:  {fid_transport:.2f}")
    print(f"  VAE Generation:  {fid_vae:.2f}")
    print("=" * 70)

    return fid_transport, fid_vae


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_transport_figure(
    implicit_net,
    cvae_source,
    cvae_target,
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype,
    epoch: int,
    loss: float,
    save_path: str,
    n_show: int = 8,
) -> None:
    """Save a 3-row grid: source | transported | target."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    implicit_net.eval()
    c_onehot = F.one_hot(labels[:n_show], num_classes).to(dtype)

    with torch.no_grad():
        x_lat, _       = cvae_source.encode(x_src[:n_show], labels[:n_show])
        z_trans        = x_lat + implicit_net.grad_g(x_lat, c_onehot)
        x_transported  = cvae_target.decode(z_trans, labels[:n_show])

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 2, 6))
    for i in range(n_show):
        axes[0, i].imshow(x_src[i].cpu().squeeze(), cmap="gray")
        axes[0, i].set_title(f"Src:{labels[i].item()}")
        axes[0, i].axis("off")

        axes[1, i].imshow(x_transported[i].cpu().squeeze(), cmap="gray")
        axes[1, i].set_title("T(x)")
        axes[1, i].axis("off")

        axes[2, i].imshow(x_tgt[i].cpu().squeeze(), cmap="gray")
        axes[2, i].set_title(f"Tgt:{labels[i].item()}")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Source",      rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("T(x)",        rotation=0, ha="right", va="center")
    axes[2, 0].set_ylabel("Target",      rotation=0, ha="right", va="center")

    plt.suptitle(f"Epoch {epoch} | Loss = {loss:.4f}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

def compute_fid(real, fake, device="cuda", batch_size=64, dims=2048):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    def act(images):
        n = images.shape[0]
        out = np.empty((n, dims))

        for i in range(0, n, batch_size):
            batch = images[i:i+batch_size].to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            with torch.no_grad():
                pred = model(batch)[0]

            pred = torch.nn.functional.adaptive_avg_pool2d(pred, 1)
            out[i:i+batch.shape[0]] = pred.cpu().numpy().reshape(batch.shape[0], -1)

        return out

    a1 = act(real)
    a2 = act(fake)

    return calculate_frechet_distance(
        np.mean(a1, axis=0), np.cov(a1, rowvar=False),
        np.mean(a2, axis=0), np.cov(a2, rowvar=False)
    )
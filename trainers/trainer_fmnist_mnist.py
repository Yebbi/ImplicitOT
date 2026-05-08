"""
trainer.py — Training loop for class-conditional latent OT.
"""

from __future__ import annotations
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_mnist_fmnist import Config
from utils.losses import mmd_rbf, ot_dual_loss
from utils.evaluation import save_transport_figure


class LatentOTTrainer:
    """Encapsulates training, checkpointing, and per-epoch visualisation."""

    def __init__(
        self,
        cfg: Config,
        implicit_net,
        cvae_source,
        cvae_target,
        paired_loader: DataLoader,
        dtype: torch.dtype,
    ) -> None:
        self.cfg            = cfg
        self.implicit_net   = implicit_net
        self.cvae_source    = cvae_source
        self.cvae_target    = cvae_target
        self.loader         = paired_loader
        self.device         = cfg.device
        self.dtype          = dtype
        self.num_classes    = cfg.vae.num_classes
        self.t              = cfg.train

        self.optimizer = torch.optim.Adam(
            implicit_net.parameters(), lr=self.t.lr
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
        )

        self.best_loss: float = float("inf")
        self._y_warm: Optional[torch.Tensor] = None   # warm-start cache

    # ------------------------------------------------------------------
    def train(self) -> None:
        self._print_header()

        for epoch in range(1, self.t.epochs + 1):
            stop = self._run_epoch(epoch)
            if stop:
                break

        print(f"\n{'='*70}")
        print(f"Training complete!  Best loss: {self.best_loss:.4f}")
        print(f"Model saved to: {self.cfg.save_path}")
        print("=" * 70)

    # ------------------------------------------------------------------
    def _run_epoch(self, epoch: int) -> bool:
        """Run one epoch.  Returns True if training should stop early."""
        self.implicit_net.train()

        metrics = dict(
            loss=0., ot_loss=0., g_source=0., conjugate=0., mmd=0.,
            depth=0., inner_grad_norm=0., param_grad_norm=0.
        )
        n_batches = 0
        start = time.time()

        pbar = tqdm(self.loader, total=len(self.loader), leave=False)
        for x_src, x_tgt, labels in pbar:
            x_src    = x_src.to(device=self.device, dtype=self.dtype)
            x_tgt    = x_tgt.to(device=self.device, dtype=self.dtype)
            labels   = labels.to(self.device)
            c_onehot = F.one_hot(labels, self.num_classes).to(self.dtype)

            # ---- encode both domains ----
            with torch.no_grad():
                x, _ = self.cvae_source.encode(x_src, labels)   # source latents
                z, _ = self.cvae_target.encode(x_tgt, labels)    # target latents

            # ---- Term 1: E_mu[g(x, k)] ----
            xc       = torch.cat([x, c_onehot], dim=1)
            g_src    = self.implicit_net.g_net(xc)               # (B, 1)

            # ---- Term 2: implicit solve on target latents ----
            y_star, depth, inner_gnorm = self.implicit_net(
                z, c_onehot,
                tol=self.t.tol,
                max_iter=self.t.max_iters,
                verbose=False,
                y_init=self._y_warm,
            )
            self._y_warm = y_star.detach()

            yc      = torch.cat([y_star.detach(), c_onehot], dim=1)
            g_ystar = self.implicit_net.g_net(yc)                # (B, 1)

            # ---- OT dual loss ----
            loss_ot = ot_dual_loss(g_src, g_ystar)

            # ---- MMD regulariser ----
            loss_mmd = torch.zeros((), device=self.device, dtype=self.dtype)
            if self.t.lambda_mmd > 0:
                n_sub = min(self.t.mmd_subsample, x.size(0))
                idx   = torch.randperm(x.size(0), device=self.device)[:n_sub]
                grad_gx       = self.implicit_net.grad_g(x[idx], c_onehot[idx], create_graph=True)
                z_transported = x[idx] + grad_gx
                loss_mmd      = mmd_rbf(z_transported, z[idx])

            loss = loss_ot + self.t.lambda_mmd * loss_mmd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- accumulate metrics ----
            metrics["loss"]           += loss.item()
            metrics["ot_loss"]        += loss_ot.item()
            metrics["g_source"]       += g_src.mean().item()
            metrics["conjugate"]      += g_ystar.mean().item()
            metrics["mmd"]            += loss_mmd.item()
            metrics["depth"]           = max(metrics["depth"], depth)
            metrics["inner_grad_norm"] = max(metrics["inner_grad_norm"], inner_gnorm)
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "OT":   f"{loss_ot.item():.4f}",
                "MMD":  f"{loss_mmd.item():.2e}",
                "depth": f"{depth:.1f}",
            })

        # ---- epoch-level aggregation ----
        for k in ("loss", "ot_loss", "g_source", "conjugate", "mmd"):
            metrics[k] /= n_batches

        metrics["param_grad_norm"] = torch.sqrt(sum(
            p.grad.detach().pow(2).sum()
            for p in self.implicit_net.parameters()
            if p.grad is not None
        )).item()

        self.scheduler.step(metrics["loss"])
        elapsed = time.time() - start

        # ---- early stopping ----
        if metrics["inner_grad_norm"] > self.t.inner_grad_norm_threshold:
            print(
                f"Epoch {epoch}: inner_grad_norm={metrics['inner_grad_norm']:.2e} "
                f"> {self.t.inner_grad_norm_threshold} — stopping early (weights NOT saved)"
            )
            return True

        # ---- checkpoint ----
        saved = 0
        if metrics["loss"] < self.best_loss:
            self.best_loss = metrics["loss"]
            os.makedirs(os.path.dirname(self.cfg.save_path) or ".", exist_ok=True)
            torch.save(self.implicit_net.state_dict(), self.cfg.save_path)
            saved = 1

        # ---- logging ----
        if epoch % self.t.print_freq == 0:
            log = (
                f"Epoch {epoch:3d}/{self.t.epochs} | "
                f"loss {metrics['loss']:.4f} | "
                f"OT {metrics['ot_loss']:.4f} | "
                f"E[g(x)] {metrics['g_source']:.4f} | "
                f"E[conj] {metrics['conjugate']:.4f} | "
            )
            if self.t.lambda_mmd > 0:
                log += f"MMD {metrics['mmd']:.2e} | "
            log += (
                f"depth {metrics['depth']:.1f} | "
                f"inner_gnorm {metrics['inner_grad_norm']:.2e} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.1e} | "
                f"saved {saved} | "
                f"time {elapsed:.1f}s | "
                f"net_gnorm {metrics['param_grad_norm']:.2e}"
            )
            print(log)

        # ---- visualisation ----
        if epoch % self.t.plot_freq == 0:
            save_transport_figure(
                implicit_net=self.implicit_net,
                cvae_source=self.cvae_source,
                cvae_target=self.cvae_target,
                x_src=x_src,
                x_tgt=x_tgt,
                labels=labels,
                num_classes=self.num_classes,
                dtype=self.dtype,
                epoch=epoch,
                loss=metrics["loss"],
                save_path=self.cfg.save_fig_path,
            )

        return False   # continue training

    # ------------------------------------------------------------------
    def _print_header(self) -> None:
        t = self.t
        print("=" * 70)
        print("TRAINING — CLASS-CONDITIONAL OT DUAL LOSS")
        if t.lambda_mmd > 0:
            print(
                f"+ MMD REGULARIZER in LATENT SPACE "
                f"(lambda={t.lambda_mmd}, subsample={t.mmd_subsample})"
            )
        print("=" * 70)
        print(f"Source: {self.cfg.data.source_dataset}  |  "
              f"Target: {self.cfg.data.target_dataset}")
        print(f"Latent dim: {self.cfg.vae.latent_dim}  |  "
              f"Classes: {self.num_classes}")
        print("=" * 70 + "\n")

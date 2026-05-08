"""
trainer.py — Training loop for Gaussian OT with ImplicitICNN.
"""

from __future__ import annotations
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

from configs.config_gaussians import Config
from utils.losses import ot_dual_loss
from utils.true_gaussians import compute_ot_map_matrix, compute_uvp
from utils.general import compute_grad_norm, append_log, mkdir_ifnotexists


class GaussianOTTrainer:
    """Encapsulates training, checkpointing, and logging for Gaussian OT."""

    def __init__(
        self,
        cfg: Config,
        model,
        dataset,
    ) -> None:
        self.cfg     = cfg
        self.model   = model
        self.dataset = dataset
        self.device  = cfg.device
        self.t       = cfg.train
        self.s       = cfg.solver

        # resolve learning rate
        cfg.resolve()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.t.lr)

        # pre-compute ground-truth OT map if feasible
        self._measure_error = cfg.data.dim < 200
        if self._measure_error:
            self.Gamma     = compute_ot_map_matrix(dataset.Sigma0, dataset.Sigma1)
            self.Gamma_inv = np.linalg.inv(self.Gamma).real
        else:
            self.Gamma = self.Gamma_inv = None

        mkdir_ifnotexists(cfg.exp_dir())

        # write header
        append_log(
            cfg.log_path(),
            f"{cfg.data.dim}D | Var0: {dataset.var0}, Var1: {dataset.var1}\n"
            f"Hyperparameters: {self._model_hparams()}\n"
            f"Number of parameters: {sum(p.numel() for p in model.parameters())}\n",
        )

        # cached batch (resampled every sampling_freq epochs)
        self._x_batch: Optional[torch.Tensor] = None
        self._z_batch: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def train(self) -> None:
        self._print_header()
        self.model.train()

        for epoch in range(self.t.max_epochs):
            self._run_epoch(epoch)

        print("Training complete.")

    # ------------------------------------------------------------------
    def _run_epoch(self, epoch: int) -> None:
        start = time.time()

        # resample data periodically
        if epoch % self.t.sampling_freq == 0:
            self._x_batch, self._z_batch = self.dataset.sample(
                self.t.batch_size, self.device
            )
        x_batch = self._x_batch
        z_batch = self._z_batch

        self.optimizer.zero_grad()

        # ---- Term 1: E_mu[g(x)] ----
        g_src = self.model.g_net(x_batch).mean()

        # ---- Term 2: fixed-point solve on target samples ----
        y_star, depth, grad_y_norm = self.model(
            z_batch,
            tol=self.s.tol,
            max_iter=self.s.max_iters,
            verbose=False,
        )

        g_ystar = (
            0.5 * (torch.norm(z_batch - y_star.detach(), dim=1) ** 2).mean()
            + self.model.g_net(y_star.detach()).mean()
        )

        # ---- OT dual loss ----
        loss_ot   = ot_dual_loss(g_src, g_ystar)
        loss_ot.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.t.grad_clip)
        self.optimizer.step()

        self.model.convexify()

        elapsed    = time.time() - start
        grad_norm  = compute_grad_norm(self.model)

        # ---- checkpoint ----
        is_last = epoch == self.t.max_epochs - 1
        if epoch % self.t.save_freq == 0 or is_last:
            ckpt_path = os.path.join(self.cfg.exp_dir(), f"model_ep{epoch}.pth")
            torch.save(self.model.state_dict(), ckpt_path)

        # ---- logging ----
        if epoch % self.t.log_freq == 0 or is_last:
            fwd_uvp, bwd_uvp = self._compute_uvp(x_batch, z_batch, y_star)

            log_str = (
                f"epoch: {epoch + 1}, "
                f"total_loss: {loss_ot.item():.3e}, "
                f"ot_loss: {loss_ot.item():.3e}, "
                f"train fwd UVP: {fwd_uvp:.3e}, "
                f"train bwd UVP: {bwd_uvp:.3e}, "
                f"depth: {depth}, "
                f"|nabla L|: {grad_norm:.3e}, "
                f"|fpt_res|: {grad_y_norm:.3e}, "
                f"time: {elapsed:.3f}s, "
                f"lr: {self.t.lr:.1e}\n"
            )
            append_log(self.cfg.log_path(), log_str)

            if epoch % (self.t.log_freq * 5) == 0:
                print(log_str, end="")

    # ------------------------------------------------------------------
    def _compute_uvp(
        self,
        x_batch: torch.Tensor,
        z_batch: torch.Tensor,
        y_star: torch.Tensor,
    ) -> Tuple[float, float]:
        if not self._measure_error:
            return -1.0, -1.0
        return compute_uvp(
            x_batch=x_batch,
            z_batch=z_batch,
            y_star_batch=y_star,
            model=self.model,
            Gamma=self.Gamma,
            Gamma_inv=self.Gamma_inv,
            var0=self.dataset.var0,
            var1=self.dataset.var1,
            dim=self.cfg.data.dim,
        )

    def _model_hparams(self) -> dict:
        m = self.cfg.model
        return {
            "dim":                self.cfg.data.dim,
            "activation":         m.activation,
            "rank":               m.rank,
            "hidden_layer_sizes": m.resolve_hidden(self.cfg.data.dim),
            "strong_convexity":   m.strong_convexity,
        }

    def _print_header(self) -> None:
        cfg = self.cfg
        print("=" * 60)
        print(f"Gaussian OT — dim={cfg.data.dim}")
        print(f"  lr={self.t.lr:.1e}  |  batch={self.t.batch_size}"
              f"  |  epochs={self.t.max_epochs}")
        print(f"  tol={self.s.tol}  |  max_iters={self.s.max_iters}")
        print(f"  measure_error={self._measure_error}")
        print(f"  log → {cfg.log_path()}")
        print("=" * 60)

# visualize_ot.py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_ot_maps(x_batch, z_batch, xhat_batch, zhat_batch, model, epoch, save_path=None):
    """
    OT map과 샘플 비교 시각화
    x_batch: mu samples (torch tensor)
    z_batch: nu samples (torch tensor)
    xhat_batch: OT 역변환된 x
    zhat_batch: OT 변환된 z
    model: 학습된 ImplicitNet 모델 (g_net 사용)
    epoch: 현재 epoch
    save_path: 저장할 경로 (None이면 저장하지 않음)
    """
    plt.figure(figsize=(15, 10))

    # ---------------- 첫번째 row ----------------
    plt.subplot(2, 3, 1)
    plt.scatter(z_batch[:, 0].detach().cpu().numpy(), z_batch[:, 1].detach().cpu().numpy(), color='deepskyblue', edgecolors='black', alpha=0.5, label='$z$')
    plt.scatter(zhat_batch[:, 0].detach().cpu().numpy(), zhat_batch[:, 1].detach().cpu().numpy(), color='tomato', edgecolors='black', alpha=1, label='$\hat{z} = T(x)$')
    plt.title(f'Epoch {epoch+1}: $\hat{{z}}$ vs $z$')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.subplot(2, 3, 2)
    plt.scatter(x_batch[:, 0].detach().cpu().numpy(), x_batch[:, 1].detach().cpu().numpy(), color='tomato', edgecolors='black', alpha=0.5, label='$x$')
    plt.scatter(xhat_batch[:, 0].detach().cpu().numpy(), xhat_batch[:, 1].detach().cpu().numpy(), color='deepskyblue', edgecolors='black', alpha=1, label='$\hat{x} = T^{-1}(z)$')
    plt.title(f'Epoch {epoch+1}: $\hat{{x}}$ vs $x$')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    # ---------------- Learned g(x) ----------------
    grid_size = 100
    x = np.linspace(-6, 6, grid_size)
    y = np.linspace(-6, 6, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float64, device=x_batch.device)
        g_values = model.g_net(grid_tensor).detach().cpu().numpy()
        G = g_values.reshape(grid_size, grid_size)
    plt.subplot(2, 3, 3)
    plt.contourf(X, Y, G, levels=50, cmap='viridis')
    plt.colorbar(label='g(x)')
    plt.title('Learned g(x)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    # ---------------- OT maps ----------------
    step = 20
    plt.subplot(2, 3, 4)
    x_np = x_batch.detach().cpu().numpy()
    zhat_np = zhat_batch.detach().cpu().numpy()
    plt.scatter(x_np[:, 0], x_np[:, 1], color='tomato', edgecolors='black', s=50, alpha=0.5)
    plt.scatter(zhat_np[:, 0], zhat_np[:, 1], color='deepskyblue', edgecolors='black', s=40, alpha=1)
    mid_points = (x_np[::step] + zhat_np[::step]) / 2
    vecs = zhat_np[::step] - x_np[::step]
    plt.quiver(mid_points[:, 0], mid_points[:, 1], vecs[:, 0], vecs[:, 1],
               color='black', width=0.005, scale=20, alpha=1)
    plt.title('OT map: x -> zhat')
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.subplot(2, 3, 5)
    z_np = z_batch.detach().cpu().numpy()
    xhat_np = xhat_batch.detach().cpu().numpy()
    plt.scatter(z_np[:, 0], z_np[:, 1], color='deepskyblue', edgecolors='black', s=50, alpha=0.5)
    plt.scatter(xhat_np[:, 0], xhat_np[:, 1], color='tomato', edgecolors='black', s=40, alpha=1)
    mid_points_inv = (z_np[::step] + xhat_np[::step]) / 2
    vecs_inv = xhat_np[::step] - z_np[::step]
    plt.quiver(mid_points_inv[:, 0], mid_points_inv[:, 1], vecs_inv[:, 0], vecs_inv[:, 1],
               color='black', width=0.005, scale=20, alpha=1)
    plt.title('OT map: z -> xhat')
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

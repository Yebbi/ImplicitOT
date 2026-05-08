# Gaussian OT — ImplicitICNN-based Optimal Transport

## Project structure

```
gaussian_ot/
├── main.py               # Entry point
├── trainer.py            # GaussianOTTrainer class
├── configs/
│   ├── __init__.py
│   └── config.py         # All hyperparameters (dataclasses)
├── utils/
│   ├── __init__.py
│   ├── dataset.py        # GaussianDataset (load .npz + sample batches)
│   ├── losses.py         # ot_dual_loss
│   ├── metrics.py        # compute_ot_map_matrix, compute_uvp
│   └── general.py        # mkdir, set_seed, logging helpers
└── models/
    ├── __init__.py
    └── ImplicitICNN.py   # (your existing file — place here)
```

## Quick start

```bash
# Default (dim=32)
python main.py

# Custom dimension and solver settings
python main.py --dim 64 --tol 1e-4 --max_iter 2000 --gpu 1

# Override learning rate explicitly (disables auto-scaling)
python main.py --dim 128 --lr 5e-5 --epochs 3000
```

## Key CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--dim` | `32` | Distribution dimensionality |
| `--epochs` | `5000` | Training epochs |
| `--batch_size` | `2048` | Mini-batch size |
| `--lr` | auto | Learning rate (1e-3 if dim<16, 1e-4 if dim≥16) |
| `--tol` | `1e-3` | Fixed-point iteration tolerance |
| `--max_iter` | `1000` | Max fixed-point iterations |
| `--gpu` | `0` | CUDA device index |
| `--data_dir` | see config | Directory with `{dim}D_Test_Distributions.npz` |
| `--output_dir` | `./exps/...` | Root directory for logs and checkpoints |

## Data format

Each `.npz` file must contain:
- `Sigma0`: source covariance matrix `(dim, dim)`
- `Sigma1`: target covariance matrix `(dim, dim)`

UVP error against the ground-truth linear OT map is computed automatically
when `dim < 200`.

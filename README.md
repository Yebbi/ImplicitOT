# Implicit OT — Fixed-Point-Based Implicit Optimal Transport

This repository implements **Implicit Optimal Transport (Implicit OT)**, a single-network formulation of quadratic optimal transport based on a proximal fixed-point reformulation of the Kantorovich dual problem.

Unlike prior approaches that rely on adversarial min–max optimization, auxiliary networks, or ODE/SDE solvers, our method enforces dual feasibility through a **proximal optimality condition**, enabling stable and scalable training using a single neural potential.

---

## 🧠 Key Features

- Single-network optimal transport (no adversarial training)
- Proximal fixed-point formulation of the Kantorovich dual
- No implicit differentiation through inner optimization
- Simultaneous recovery of forward and backward transport maps
- Supports Gaussian, image, and physics-based datasets
- Stable training in high-dimensional regimes

---

## 📁 Project Structure
## Project structure
```bash
implicit_ot/
├── run_gausssians.py     # Entry point of Gaussian OT
├── run_mnist_fmnist.py   # Entry point of image class-conditional OT
├── trainer.py            # GaussianOTTrainer class
├── configs/
│   ├── config_gaussians.py # All hyperparameters for Gaussians
│   └── config.py         # All hyperparameters for image CCOT
├── data # Data loader (images, Gaussians, physics)
├── trainers/
│   ├──trainer_gaussians.py    # Trainer for Gaussians
│   ├── trainer_cvae.py         # Trainer for conditional VAE (images)
│   └── trainer_fmnist_mnist.py # Trainer for image CCOT
├── utils/
│   ├── evaluation.py     # metrics
│   ├── losses.py         # ot_dual_loss, vae loss
│   ├── general.py        # general (seed, mkdir)
│   └── true_gaussians.py # Gaussian true OT map, UVP
└── models/
    ├── ImplicitICNN.py   # convex networks
    ├── ImplicitICNN.py   # scalar networks
    └── cvae.py           # conditional vae networks
```


---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```
Recommended environment:

* Python ≥ 3.9
* PyTorch ≥ 2.2
* CUDA 11.8 / 12.1


---

## 🚀 Quick Start
## 📌 Gaussian Optimal Transport
```bash
# Default configuration (dim=32)
python run_gaussians.py
```

```bash
# Custom dimension and solver settings
python run_gaussians.py --dim 64 --tol 1e-4 --max_iter 2000 --gpu 1
```

```bash
# Override learning rate explicitly
python run_gaussians.py --dim 128 --lr 5e-5 --epochs 3000
```


## 🖼 Class-Conditional Optimal Transport (Images)

```bash
# Default: FMNIST → MNIST
python run_mnist_fmnist.py
```

```bash
# Custom configuration
python run_mnist_fmnist.py \
    --source fmnist \
    --target mnist \
    --latent_dim 15 \
    --epochs 30 \
    --lr 1e-4 \
    --lambda_mmd 1e-3 \
    --mmd_subsample 5000 \
    --device cuda:0 \
    --precision float32 \
    --output_dir outputs/run1
```

```bash
# Skip initial FID evaluation (faster debugging)
python run_mnist_fmnist.py --skip_pre_fid
```

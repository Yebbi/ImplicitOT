import torch
from models.cvae import ConditionalConvVAE
import argparse
from data.cvae import get_loader
from trainers.trainer_cvae import train_cvae
from utils.evaluation import compute_fid

from utils.losses import vae_loss

# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="mnist")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--beta_max", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)

    return parser.parse_args()


# =========================
# Main
# =========================
def main():

    args = parse_args()

    device = args.device
    dataset_name = args.dataset

    print("=" * 60)
    print("EXPERIMENT CONFIG")
    print(f"device        : {device}")
    print(f"dataset       : {dataset_name}")
    print(f"latent_dim    : {args.latent_dim}")
    print(f"epochs        : {args.epochs}")
    print(f"lr            : {args.lr}")
    print("=" * 60)

    # -------------------------
    # Data
    # -------------------------
    loader = get_loader(
        name=dataset_name,
        batch_size=args.batch_size,
        img_size=32
    )

    # -------------------------
    # Model
    # -------------------------
    model = ConditionalConvVAE(
        latent_dim=args.latent_dim,
        num_classes=10
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # -------------------------
    # Train
    # -------------------------
    train_cvae(
        model=model,
        loader=loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        beta_max=args.beta_max,
        warmup_epochs=args.warmup_epochs,
        save_path=f"ckpts/cvae_{dataset_name}_latent{args.latent_dim}.pth",
        save_fig=f"samples_{dataset_name}.png"
    )

    # -------------------------
    # Final eval (optional)
    # -------------------------
    model.eval()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
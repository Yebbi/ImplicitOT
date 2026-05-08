"""
Class-paired dataset for cross-domain image translation.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor

from configs.config_mnist_fmnist import DataConfig


class PairedClassDataset(Dataset):
    """Yields (source_image, target_image, label) triples sharing the same class.

    At each __getitem__ call a class is sampled uniformly, then one image is
    drawn at random from each domain for that class.  This ensures the policy
    is class-balanced without pre-computing fixed pairs.
    """

    def __init__(self, cfg: DataConfig) -> None:
        tfm = Compose([Resize((cfg.img_size, cfg.img_size)), ToTensor()])

        src_ds = self._load_dataset(cfg.source_dataset, tfm)
        tgt_ds = self._load_dataset(cfg.target_dataset, tfm)

        self.src_by_class = self._group_by_class(src_ds)
        self.tgt_by_class = self._group_by_class(tgt_ds)
        self.classes = sorted(set(self.src_by_class) & set(self.tgt_by_class))

        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.length = min(len(src_ds), len(tgt_ds))

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        cls = self.classes[torch.randint(len(self.classes), (1,)).item()]

        src_idx = self._random_pick(self.src_by_class[cls])
        tgt_idx = self._random_pick(self.tgt_by_class[cls])

        src_img, _ = self.src_ds[src_idx]
        tgt_img, _ = self.tgt_ds[tgt_idx]
        return src_img, tgt_img, cls

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset(name: str, tfm):
        name = name.lower()
        if name in ("fmnist", "fashionmnist"):
            return torchvision.datasets.FashionMNIST(
                "./data", train=True, download=True, transform=tfm
            )
        if name == "mnist":
            return torchvision.datasets.MNIST(
                "./data", train=True, download=True, transform=tfm
            )
        raise ValueError(f"Unknown dataset: {name!r}. Choose 'mnist' or 'fmnist'.")

    @staticmethod
    def _group_by_class(ds) -> dict:
        groups: dict = {}
        for idx, (_, label) in enumerate(ds):
            groups.setdefault(label, []).append(idx)
        return groups

    @staticmethod
    def _random_pick(indices: list) -> int:
        return indices[torch.randint(len(indices), (1,)).item()]


def build_loader(cfg: DataConfig) -> DataLoader:
    dataset = PairedClassDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

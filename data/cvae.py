import torchvision
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

def get_loader(name: str, batch_size: int = 128, img_size: int = 32):

    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
    ])

    if name.lower() == "mnist":
        ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)

    elif name.lower() in ["fmnist", "fashionmnist"]:
        ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
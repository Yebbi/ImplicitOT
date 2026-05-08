import torch
import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils.losses import vae_loss

def train_cvae(model, loader, device, epochs, lr, beta_max,
               warmup_epochs, save_path, save_fig):

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")

    for ep in range(1, epochs+1):

        model.train()
        start = time.time()

        beta = beta_max * ep / warmup_epochs if ep <= warmup_epochs else beta_max

        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            x_hat, mu, logvar = model(x, y)
            loss, r, k = vae_loss(x, x_hat, mu, logvar, beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / len(loader)

        if avg < best:
            best = avg
            torch.save(model.state_dict(), save_path)

        print(f"[{ep}] loss={avg:.3f} beta={beta:.3f} time={time.time()-start:.2f}s")

        # optional plotting omitted for brevity
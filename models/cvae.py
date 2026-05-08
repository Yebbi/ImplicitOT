import torch
import torch.nn as nn
import torch.nn.functional as F
# ================================
# Conditional VAE (CVAE) Implementation
# ================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm for the decoder."""
    def __init__(self, channels, num_groups=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(num_groups, channels),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class ConditionalConvVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder (BatchNorm is fine here — consistent statistics)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),      # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),    # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),   # 8 -> 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),   # 4 -> 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(512*2*2 + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(512*2*2 + num_classes, latent_dim)

        # Decoder — GroupNorm instead of BatchNorm to avoid train/eval mismatch
        self.decoder_fc = nn.Linear(latent_dim + num_classes, 512*2*2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 2 -> 4
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 -> 16
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),     # 16 -> 32
            nn.Sigmoid()
        )
    
    def encode(self, x, labels):
        h = self.encoder(x)
        labels_onehot = F.one_hot(labels, self.num_classes).to(self.fc_mu.weight.dtype)
        h = torch.cat([h, labels_onehot], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z, labels):
        labels_onehot = F.one_hot(labels, self.num_classes).to(self.decoder_fc.weight.dtype)
        z_cond = torch.cat([z, labels_onehot], dim=1)
        h = self.decoder_fc(z_cond)
        h = h.view(-1, 512, 2, 2)
        return self.decoder(h)
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, labels)
        return x_hat, mu, logvar
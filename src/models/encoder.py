"""Convolutional encoder and decoder for VAE / VQ-VAE."""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Downsampling encoder using stride-2 convolutions.

    Input:  (N, in_channels, H, W)
    Output: (N, flat_dim) where flat_dim = last_channels * (H / 2^n_layers) * (W / 2^n_layers)
    """

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(),
            ])
            ch_in = ch_out
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Upsampling decoder using stride-2 transposed convolutions.

    Input:  (N, first_channels, H_small, W_small)
    Output: (N, out_channels, H, W) with sigmoid activation
    """

    def __init__(self, channels: list[int], out_channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
            ])
        # Final layer to image channels
        layers.append(
            nn.ConvTranspose2d(
                channels[-1], out_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            )
        )
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAEModel(nn.Module):
    """Convolutional VAE: encoder → (mu, log_var) → reparameterize → decoder."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(in_channels, encoder_channels)

        # Compute spatial size after encoder downsampling
        n_downsample = len(encoder_channels)
        self.enc_spatial = image_size // (2 ** n_downsample)
        enc_flat = encoder_channels[-1] * self.enc_spatial * self.enc_spatial

        self.fc_mu = nn.Linear(enc_flat, latent_dim)
        self.fc_log_var = nn.Linear(enc_flat, latent_dim)

        # Decoder projection
        dec_first = decoder_channels[0]
        self.dec_spatial = self.enc_spatial
        self.fc_decode = nn.Linear(latent_dim, dec_first * self.dec_spatial * self.dec_spatial)
        self.dec_first = dec_first

        self.decoder = ConvDecoder(decoder_channels, out_channels=in_channels)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.dec_first, self.dec_spatial, self.dec_spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

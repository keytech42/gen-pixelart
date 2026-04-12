"""Shared model building blocks: ResBlock with optional time conditioning."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with optional time embedding injection.

    If time_emb_dim is provided, a time embedding vector is projected
    and added to the feature map between the two conv layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_proj = None

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        if self.time_proj is not None and t_emb is not None:
            # Project time embedding and add to feature map
            t = self.act(self.time_proj(t_emb))
            h = h + t[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample(nn.Module):
    """Spatial downsampling by 2x using stride-2 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by 2x using nearest interpolation + convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps.

    Maps integer timestep → high-dimensional embedding via sin/cos,
    then projects through an MLP.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

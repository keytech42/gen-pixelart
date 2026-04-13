"""Lightweight U-Net for diffusion denoising."""

import torch
import torch.nn as nn

from src.models.blocks import Downsample, ResBlock, SinusoidalTimeEmbedding, Upsample


class UNet(nn.Module):
    """U-Net with time conditioning for DDPM.

    Architecture:
      - Input conv
      - Down path: (ResBlock * num_res_blocks + Downsample) per level
      - Middle: ResBlock + ResBlock
      - Up path: (ResBlock * num_res_blocks + Upsample) per level, with skip connections
      - Output conv

    No attention — overkill for 16x16 images.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: list[int],
        num_res_blocks: int,
        time_emb_dim: int,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        # Optional class conditioning — added to time embedding
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_emb_dim)
        else:
            self.class_embed = None

        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i, ch in enumerate(channels):
            ch_in = channels[i - 1] if i > 0 else channels[0]
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(ch_in if j == 0 else ch, ch, time_emb_dim))
            self.down_blocks.append(blocks)
            if i < len(channels) - 1:
                self.down_samples.append(Downsample(ch))
            else:
                self.down_samples.append(nn.Identity())

        # Middle
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim)

        # Up path (reversed channels, with skip connections doubling input channels)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i, ch in enumerate(rev_channels):
            ch_out = rev_channels[i + 1] if i < len(rev_channels) - 1 else rev_channels[-1]
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # First block takes skip connection (double channels)
                if j == 0:
                    blocks.append(ResBlock(ch * 2, ch_out, time_emb_dim))
                else:
                    blocks.append(ResBlock(ch_out, ch_out, time_emb_dim))
            self.up_blocks.append(blocks)
            if i < len(rev_channels) - 1:
                self.up_samples.append(Upsample(ch_out))
            else:
                self.up_samples.append(nn.Identity())

        # Output
        self.out_norm = nn.GroupNorm(8, rev_channels[-1])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(rev_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_label: torch.Tensor | None = None) -> torch.Tensor:
        t_emb = self.time_embed(t)

        # Add class conditioning to time embedding
        if class_label is not None and self.class_embed is not None:
            t_emb = t_emb + self.class_embed(class_label)

        h = self.input_conv(x)

        # Down path — collect skip connections
        skips = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if not isinstance(downsample, nn.Identity):
                h = downsample(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Up path — consume skip connections in reverse
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            if not isinstance(upsample, nn.Identity):
                h = upsample(h)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)

"""VQ-VAE strategy — encoder → vector quantization → decoder."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.codebook import VectorQuantizer
from src.models.encoder import ConvDecoder, ConvEncoder
from src.models.prior import IndexPrior
from src.strategies.base import GenerativeStrategy


class VQVAEModel(nn.Module):
    """VQ-VAE: encoder → quantize spatial features → decoder."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        codebook_size: int,
        codebook_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_channels, encoder_channels)
        self.quantizer = VectorQuantizer(codebook_size, codebook_dim)
        self.latent_spatial = image_size // (2 ** len(encoder_channels))
        self.decoder = ConvDecoder(decoder_channels, out_channels=in_channels)

        # Project encoder output to codebook dim if they don't match
        enc_out = encoder_channels[-1]
        if enc_out != codebook_dim:
            self.pre_quant = nn.Conv2d(enc_out, codebook_dim, kernel_size=1)
            self.post_quant = nn.Conv2d(codebook_dim, decoder_channels[0], kernel_size=1)
        else:
            self.pre_quant = nn.Identity()
            self.post_quant = nn.Identity() if codebook_dim == decoder_channels[0] else nn.Conv2d(codebook_dim, decoder_channels[0], kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_e = self.pre_quant(z_e)
        z_q, indices, codebook_loss, commitment_loss = self.quantizer(z_e)
        return z_q, indices, codebook_loss, commitment_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z_q = self.post_quant(z_q)
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, indices, codebook_loss, commitment_loss = self.encode(x)
        recon = self.decode(z_q)
        return recon, indices, codebook_loss, commitment_loss


class VQVAEStrategy(GenerativeStrategy):
    """VQ-VAE generative strategy.

    Loss: recon + codebook_loss + commitment_weight * commitment_loss
    Per the original VQ-VAE paper, codebook loss always has weight 1.0.
    """

    def __init__(self) -> None:
        self.commitment_weight: float = 0.25
        self.prior: IndexPrior | None = None

    def build_model(self, config: DictConfig) -> nn.Module:
        self.commitment_weight = config.model.commitment_weight
        return VQVAEModel(
            in_channels=3,
            image_size=config.dataset.image_size,
            encoder_channels=list(config.model.encoder_channels),
            decoder_channels=list(config.model.decoder_channels),
            codebook_size=config.model.codebook_size,
            codebook_dim=config.model.codebook_dim,
        )

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, float]:
        optimizer.zero_grad()
        recon, indices, codebook_loss, commitment_loss = model(batch)

        recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction="mean")
        # Paper: L = recon + ||sg[z_e] - e||² + β * ||z_e - sg[e]||²
        # codebook_loss always weight 1.0, only commitment_loss gets β
        loss = recon_loss + codebook_loss + self.commitment_weight * commitment_loss

        loss.backward()
        optimizer.step()

        # Codebook utilization: fraction of codebook entries used in this batch
        n_used = indices.unique().numel()
        n_total = model.quantizer.num_embeddings
        utilization = n_used / n_total

        return {
            "recon_loss": recon_loss.item(),
            "codebook_loss": codebook_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "total_loss": loss.item(),
            "codebook_utilization": utilization,
        }

    def sample(
        self,
        model: nn.Module,
        n_samples: int,
        device: torch.device,
        class_label: int | None = None,
    ) -> torch.Tensor:
        spatial = model.latent_spatial
        if self.prior is not None:
            # Sample from learned prior
            flat_indices = self.prior.sample(n_samples, device)
            indices = flat_indices.view(n_samples, spatial, spatial)
        else:
            # Fallback: random indices (incoherent but functional)
            indices = torch.randint(
                0, model.quantizer.num_embeddings,
                (n_samples, spatial, spatial),
                device=device,
            )
        z_q = model.quantizer.decode_indices(indices)
        return model.decode(z_q)

    def get_metrics(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            recon, indices, codebook_loss, commitment_loss = model(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch).item()
            n_used = indices.unique().numel()
            n_total = model.quantizer.num_embeddings
        return {
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "codebook_utilization": n_used / n_total,
        }

"""VAE strategy — convolutional VAE with reparameterization trick."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.encoder import VAEModel
from src.strategies.base import GenerativeStrategy


class VAEStrategy(GenerativeStrategy):
    """VAE generative strategy.

    Loss: reconstruction (MSE or BCE) + beta-weighted KL divergence.
    Use BCE for binary sprites, MSE for colored/continuous.
    """

    def __init__(self) -> None:
        self.kl_weight: float = 0.0005
        self.recon_loss_type: str = "mse"

    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.recon_loss_type == "bce":
            return F.binary_cross_entropy(recon, target, reduction="mean")
        return F.mse_loss(recon, target, reduction="mean")

    def build_model(self, config: DictConfig) -> nn.Module:
        self.kl_weight = config.model.kl_weight
        self.recon_loss_type = config.model.get("recon_loss", "mse")
        return VAEModel(
            in_channels=3,
            image_size=config.dataset.image_size,
            encoder_channels=list(config.model.encoder_channels),
            decoder_channels=list(config.model.decoder_channels),
            latent_dim=config.model.latent_dim,
        )

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, float]:
        optimizer.zero_grad()
        recon, mu, log_var = model(batch)

        recon_loss = self._recon_loss(recon, batch)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recon_loss + self.kl_weight * kl_loss

        loss.backward()
        optimizer.step()

        return {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": loss.item(),
        }

    def sample(
        self,
        model: nn.Module,
        n_samples: int,
        device: torch.device,
        class_label: int | None = None,
    ) -> torch.Tensor:
        z = torch.randn(n_samples, model.latent_dim, device=device)
        return model.decode(z)

    def get_metrics(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            recon, mu, log_var = model(batch)
            recon_loss = self._recon_loss(recon, batch).item()
            kl_loss = (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())).item()
        return {"recon_loss": recon_loss, "kl_loss": kl_loss}

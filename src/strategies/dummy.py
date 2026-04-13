"""Dummy strategy for testing the training pipeline end-to-end."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.strategies.base import GenerativeStrategy


class DummyModel(nn.Module):
    """Minimal model that maps input to output of same shape."""

    def __init__(self, image_size: int, channels: int = 3) -> None:
        super().__init__()
        flat_dim = channels * image_size * image_size
        self.linear = nn.Linear(flat_dim, flat_dim)
        self.image_size = image_size
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        out = self.linear(flat)
        return out.reshape(batch_size, self.channels, self.image_size, self.image_size)


class DummyStrategy(GenerativeStrategy):
    """Dummy strategy that produces random-ish output. For pipeline testing only."""

    def build_model(self, config: DictConfig) -> nn.Module:
        image_size = config.dataset.image_size
        return DummyModel(image_size=image_size)

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, float]:
        optimizer.zero_grad()
        out = model(batch)
        loss = torch.mean((out - batch) ** 2)
        loss.backward()
        optimizer.step()
        return {"mse_loss": loss.item()}

    def sample(
        self,
        model: nn.Module,
        n_samples: int,
        device: torch.device,
        class_label: int | None = None,
    ) -> torch.Tensor:
        image_size = model.image_size  # type: ignore[attr-defined]
        return torch.rand(n_samples, 3, image_size, image_size, device=device)

    def get_metrics(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            out = model(batch)
            mse = torch.mean((out - batch) ** 2).item()
        return {"mse": mse}

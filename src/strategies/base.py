"""GenerativeStrategy ABC — the strategy contract for all generative architectures."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from omegaconf import DictConfig


class GenerativeStrategy(ABC):
    """Abstract base class for generative strategies.

    Each strategy encapsulates model construction, a single training step,
    sampling, and metric computation. The Trainer (context) calls these
    methods without knowing which generative approach is running.

    Labels are optional throughout — when provided, strategies that support
    conditioning use them; others ignore them.
    """

    @abstractmethod
    def build_model(self, config: DictConfig) -> nn.Module:
        """Construct and return the model."""
        ...

    @abstractmethod
    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one training step. Returns a loss dict with named losses."""
        ...

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        n_samples: int,
        device: torch.device,
        class_label: int | None = None,
    ) -> torch.Tensor:
        """Generate samples. Returns tensor of shape (N, C, H, W).

        If class_label is provided and the strategy supports conditioning,
        generates samples of that class.
        """
        ...

    @abstractmethod
    def get_metrics(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics on a batch."""
        ...

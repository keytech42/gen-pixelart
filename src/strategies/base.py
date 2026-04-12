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
    ) -> dict[str, float]:
        """Run one training step. Returns a loss dict with named losses."""
        ...

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        n_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate samples. Returns tensor of shape (N, C, H, W)."""
        ...

    @abstractmethod
    def get_metrics(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics on a batch."""
        ...

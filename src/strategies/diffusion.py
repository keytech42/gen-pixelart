"""DDPM diffusion strategy — iterative denoising from pure noise."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.unet import UNet
from src.strategies.base import GenerativeStrategy


class NoiseScheduler:
    """Linear beta noise schedule for DDPM.

    Precomputes all the alpha/alpha_bar constants needed for
    the forward (q_sample) and reverse (p_sample) processes.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | None = None,
    ) -> None:
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Store all precomputed values
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])) / (1.0 - alpha_bar)

        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "NoiseScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_recip_alpha = self.sqrt_recip_alpha.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise to x_0 at timestep t.

        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        return x_t, noise

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse process: denoise x_t by one step.

        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta) + sigma_t * z
        """
        t_idx = t[0].item()  # All elements in batch have same t during sampling

        pred_noise = model(x_t, t)

        beta = self.betas[t_idx]
        sqrt_recip_alpha = self.sqrt_recip_alpha[t_idx]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t_idx]

        # Predicted mean
        mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_ab * pred_noise)

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t_idx])
            return mean + sigma * noise
        else:
            return mean


class DiffusionStrategy(GenerativeStrategy):
    """DDPM generative strategy.

    Training: sample timestep, add noise, predict noise, MSE loss.
    Sampling: iterative denoising from pure noise over T steps.
    """

    def __init__(self) -> None:
        self.scheduler: NoiseScheduler | None = None
        self.timesteps: int = 1000
        self.image_size: int = 16

    def build_model(self, config: DictConfig) -> nn.Module:
        self.timesteps = config.model.timesteps
        self.image_size = config.dataset.image_size
        self.scheduler = NoiseScheduler(
            timesteps=self.timesteps,
            beta_start=config.model.beta_start,
            beta_end=config.model.beta_end,
        )
        return UNet(
            in_channels=3,
            out_channels=3,
            channels=list(config.model.channels),
            num_res_blocks=config.model.num_res_blocks,
            time_emb_dim=config.model.time_emb_dim,
        )

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: torch.Tensor,
    ) -> dict[str, float]:
        assert self.scheduler is not None
        device = batch.device

        # Move scheduler to device on first call
        if self.scheduler.betas.device != device:
            self.scheduler.to(device)

        optimizer.zero_grad()

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch.shape[0],), device=device)

        # Forward process: add noise
        x_t, noise = self.scheduler.q_sample(batch, t)

        # Predict the noise
        pred_noise = model(x_t, t)

        # Loss: MSE between predicted and actual noise
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        loss.backward()
        optimizer.step()

        return {"noise_prediction_mse": loss.item()}

    def sample(
        self,
        model: nn.Module,
        n_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert self.scheduler is not None
        if self.scheduler.betas.device != device:
            self.scheduler.to(device)

        # Start from pure noise
        x = torch.randn(n_samples, 3, self.image_size, self.image_size, device=device)

        # Iterative denoising
        for t_val in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
            x = self.scheduler.p_sample(model, x, t)

        # Clamp to [0, 1]
        return x.clamp(0, 1)

    def get_metrics(
        self,
        model: nn.Module,
        batch: torch.Tensor,
    ) -> dict[str, float]:
        assert self.scheduler is not None
        device = batch.device
        if self.scheduler.betas.device != device:
            self.scheduler.to(device)

        with torch.no_grad():
            t = torch.randint(0, self.timesteps, (batch.shape[0],), device=device)
            x_t, noise = self.scheduler.q_sample(batch, t)
            pred_noise = model(x_t, t)
            loss = torch.nn.functional.mse_loss(pred_noise, noise).item()

        return {"noise_prediction_mse": loss}

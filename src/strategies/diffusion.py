"""DDPM/DDIM diffusion strategy — iterative denoising from pure noise."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.unet import UNet
from src.strategies.base import GenerativeStrategy


class NoiseScheduler:
    """Linear beta noise schedule for DDPM/DDIM.

    Precomputes all the alpha/alpha_bar constants needed for
    the forward (q_sample) and reverse (p_sample / ddim_sample) processes.
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
        """DDPM reverse process: denoise x_t by one step (stochastic)."""
        t_idx = t[0].item()

        pred_noise = model(x_t, t)

        beta = self.betas[t_idx]
        sqrt_recip_alpha = self.sqrt_recip_alpha[t_idx]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t_idx]

        mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_ab * pred_noise)

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t_idx])
            return mean + sigma * noise
        else:
            return mean

    def ddim_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timestep_seq: list[int],
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling: deterministic (eta=0) or stochastic (eta>0) denoising.

        Skips timesteps by using the non-Markovian DDIM update rule.
        When eta=0, sampling is fully deterministic (same noise → same image).
        When eta=1, equivalent to DDPM.

        Args:
            model: noise prediction U-Net.
            x: (N, C, H, W) starting noise.
            timestep_seq: descending list of timesteps to denoise through.
            eta: stochasticity parameter (0 = deterministic).

        Returns:
            (N, C, H, W) denoised images.
        """
        n = x.shape[0]
        device = x.device

        for i in range(len(timestep_seq)):
            t_cur = timestep_seq[i]
            t_prev = timestep_seq[i + 1] if i + 1 < len(timestep_seq) else 0

            t_batch = torch.full((n,), t_cur, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha_bar_t = self.alpha_bar[t_cur]
            alpha_bar_prev = self.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            # Predict x_0 from x_t and predicted noise
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            # DDIM variance
            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * pred_noise

            # DDIM update
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if eta > 0 and t_prev > 0:
                x = x + sigma * torch.randn_like(x)

        return x


class DiffusionStrategy(GenerativeStrategy):
    """DDPM/DDIM generative strategy.

    Training: sample timestep, add noise, predict noise, MSE loss.
    Sampling: DDPM (1000 steps, stochastic) or DDIM (configurable steps, deterministic).
    """

    def __init__(self) -> None:
        self.scheduler: NoiseScheduler | None = None
        self.timesteps: int = 1000
        self.image_size: int = 16
        self.sampling_method: str = "ddpm"
        self.sampling_steps: int = 1000

    def build_model(self, config: DictConfig) -> nn.Module:
        self.timesteps = config.model.timesteps
        self.image_size = config.dataset.image_size
        self.sampling_method = config.model.get("sampling_method", "ddpm")
        self.sampling_steps = config.model.get("sampling_steps", self.timesteps)
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

        if self.scheduler.betas.device != device:
            self.scheduler.to(device)

        optimizer.zero_grad()

        t = torch.randint(0, self.timesteps, (batch.shape[0],), device=device)
        x_t, noise = self.scheduler.q_sample(batch, t)
        pred_noise = model(x_t, t)
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

        x = torch.randn(n_samples, 3, self.image_size, self.image_size, device=device)

        if self.sampling_method == "ddim":
            step_size = max(1, self.timesteps // self.sampling_steps)
            timestep_seq = list(range(self.timesteps - 1, 0, -step_size))
            x = self.scheduler.ddim_sample(model, x, timestep_seq, eta=0.0)
        else:
            for t_val in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
                x = self.scheduler.p_sample(model, x, t)

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

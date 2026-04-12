"""Trainer — the context class that owns the training loop.

Calls strategy methods without knowing which generative approach is running.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from omegaconf import DictConfig

from src.mlflow_logger import MLflowLogger
from src.palette import generate_default_palette, snap_to_palette
from src.strategies.base import GenerativeStrategy

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_sample_grid(
    images: torch.Tensor, path: Path, grid_cols: int = 8, scale: int = 4,
) -> None:
    """Save a grid of images as a PNG file, upscaled with nearest-neighbor."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — skipping sample grid save.")
        return

    images_np = images.detach().cpu().numpy()
    # (N, C, H, W) -> (N, H, W, C)
    if images_np.ndim == 4 and images_np.shape[1] in (1, 3):
        images_np = images_np.transpose(0, 2, 3, 1)
    images_np = np.clip(images_np * 255, 0, 255).astype(np.uint8)

    # Upscale each sprite with nearest-neighbor (preserves pixel art crispness)
    n, h, w, c = images_np.shape
    h_s, w_s = h * scale, w * scale
    images_scaled = np.zeros((n, h_s, w_s, c), dtype=np.uint8)
    for i in range(n):
        img = Image.fromarray(images_np[i])
        images_scaled[i] = np.array(img.resize((w_s, h_s), Image.NEAREST))

    grid_rows = (n + grid_cols - 1) // grid_cols
    grid = np.zeros((grid_rows * h_s, grid_cols * w_s, c), dtype=np.uint8)
    for i in range(n):
        r, col = divmod(i, grid_cols)
        grid[r * h_s : (r + 1) * h_s, col * w_s : (col + 1) * w_s] = images_scaled[i]

    if c == 1:
        grid = grid.squeeze(-1)
    Image.fromarray(grid).save(path)


class Trainer:
    """Context class for training generative strategies."""

    def __init__(
        self,
        strategy: GenerativeStrategy,
        config: DictConfig,
        strategy_name: str,
        palette: torch.Tensor | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.strategy_name = strategy_name
        self.device = _get_device()

        self.model = strategy.build_model(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
        )

        self.mlflow_logger = MLflowLogger(config, strategy_name)
        if palette is not None:
            self.palette = palette.to(self.device)
        else:
            self.palette = generate_default_palette(config.dataset.palette_size).to(self.device)

        logger.info(
            "Trainer initialized: strategy=%s image_size=%d device=%s",
            strategy_name,
            config.dataset.image_size,
            self.device,
        )

    def train(self, dataset: Dataset) -> None:
        """Run the full training loop over the dataset."""
        cfg = self.config.training
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        global_step = 0

        for epoch in range(cfg.epochs):
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)

                loss_dict = self.strategy.train_step(self.model, self.optimizer, batch)

                if global_step % cfg.log_interval == 0:
                    self.mlflow_logger.log_metrics(loss_dict, step=global_step)
                    loss_str = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                    logger.info("epoch=%d step=%d %s", epoch, global_step, loss_str)

                global_step += 1

            if (epoch + 1) % cfg.sample_interval == 0:
                self._log_samples(epoch)

        self._log_samples("final")
        self.mlflow_logger.end_run()
        logger.info("Training complete. Total steps: %d", global_step)

    def _log_samples(self, label: str | int) -> None:
        """Generate, palette-snap, and log sample images."""
        self.model.eval()
        with torch.no_grad():
            raw_samples = self.strategy.sample(self.model, n_samples=16, device=self.device)
            snapped = snap_to_palette(raw_samples, self.palette)
        self.model.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"samples_{label}.png"
            _save_sample_grid(snapped, path)
            self.mlflow_logger.log_artifact(path)
        logger.info("Logged sample grid: %s", label)

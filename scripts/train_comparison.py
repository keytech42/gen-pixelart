"""Train all three strategies on the focused dataset and generate comparison grids."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.augmentation import PixelArtAugment
from src.data.dataset import SpriteDataset, load_sprite_tensor
from src.palette import extract_palette, snap_to_palette
from src.strategies.diffusion import DiffusionStrategy
from src.strategies.vae import VAEStrategy
from src.strategies.vqvae import VQVAEStrategy
from src.config import load_config
from src.trainer import _save_sample_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EPOCHS = 1000
DATA_DIR = Path("data/sprites_colored_focused")
OUTPUT_DIR = Path("data/comparison")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_strategy(
    name: str,
    strategy_cls: type,
    config_path: str,
    dataset: SpriteDataset,
    palette: torch.Tensor,
    epochs: int,
) -> None:
    logger.info("=== Training %s for %d epochs ===", name, epochs)

    config = load_config(config_path)
    strategy = strategy_cls()
    model = strategy.build_model(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    params = sum(p.numel() for p in model.parameters())
    logger.info("%s: %d parameters", name, params)

    for epoch in range(epochs):
        epoch_losses: list[dict[str, float]] = []
        for batch in loader:
            batch = batch.to(DEVICE)
            loss_dict = strategy.train_step(model, optimizer, batch)
            epoch_losses.append(loss_dict)

        if epoch % 50 == 0 or epoch == epochs - 1:
            avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in epoch_losses[0]}
            loss_str = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
            logger.info("[%s] epoch %d/%d: %s", name, epoch, epochs, loss_str)

        if epoch % 100 == 99 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                samples = strategy.sample(model, 16, DEVICE)
                snapped = snap_to_palette(samples, palette)
            model.train()
            out_path = OUTPUT_DIR / f"{name}_epoch{epoch + 1}.png"
            _save_sample_grid(snapped, out_path, scale=4)
            logger.info("[%s] Saved samples to %s", name, out_path)

    # Final samples
    model.eval()
    with torch.no_grad():
        samples = strategy.sample(model, 16, DEVICE)
        snapped = snap_to_palette(samples, palette)
    _save_sample_grid(snapped, OUTPUT_DIR / f"{name}_final.png", scale=4)

    # Also save reconstructions for VAE/VQ-VAE
    if hasattr(model, 'encode') or hasattr(model, 'forward'):
        try:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(loader)).to(DEVICE)
                if name == "vae":
                    recon, _, _ = model(sample_batch[:16])
                elif name == "vqvae":
                    recon, _, _, _ = model(sample_batch[:16])
                else:
                    recon = None
                if recon is not None:
                    snapped_recon = snap_to_palette(recon, palette)
                    _save_sample_grid(snapped_recon, OUTPUT_DIR / f"{name}_recon.png", scale=4)
                    _save_sample_grid(
                        snap_to_palette(sample_batch[:16], palette),
                        OUTPUT_DIR / f"{name}_input.png", scale=4,
                    )
                    logger.info("[%s] Saved reconstructions", name)
        except Exception as e:
            logger.warning("[%s] Could not save reconstructions: %s", name, e)

    logger.info("=== %s training complete ===", name)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset
    augment = PixelArtAugment(flip_prob=0.5)
    dataset = SpriteDataset(DATA_DIR, image_size=16, transform=augment)
    raw_tensor = load_sprite_tensor(DATA_DIR, image_size=16)
    config = load_config("configs/vae.yaml")  # Any config — they share base.yaml
    palette = extract_palette(raw_tensor, num_colors=config.dataset.palette_size).to(DEVICE)
    logger.info("Dataset: %d sprites, palette: %s", len(dataset), palette.cpu().tolist())

    # Save real data grid for comparison
    _save_sample_grid(raw_tensor[:32], OUTPUT_DIR / "real_data.png", scale=4)

    # Train all three
    strategies = [
        ("vae", VAEStrategy, "configs/vae.yaml"),
        ("vqvae", VQVAEStrategy, "configs/vqvae.yaml"),
        ("diffusion", DiffusionStrategy, "configs/diffusion.yaml"),
    ]

    for name, cls, config_path in strategies:
        train_strategy(name, cls, config_path, dataset, palette, EPOCHS)

    logger.info("All training complete. Results in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""Two-stage VQ-VAE training: (1) train VQ-VAE, (2) train autoregressive prior.

Stage 1: Train VQ-VAE encoder/decoder on sprites (standard train_step).
Stage 2: Freeze VQ-VAE, encode dataset → codebook indices, train prior.
Then sample from prior → decode through frozen VQ-VAE.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data.augmentation import PixelArtAugment
from src.data.dataset import SpriteDataset, load_sprite_tensor
from src.models.prior import IndexPrior
from src.palette import extract_palette, snap_to_palette
from src.strategies.vqvae import VQVAEStrategy
from src.trainer import _save_sample_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = Path("data/sprites_focused")
OUTPUT_DIR = Path("data/comparison")


def stage1_train_vqvae(epochs: int = 500) -> tuple:
    """Train VQ-VAE and return model + strategy."""
    logger.info("=== Stage 1: Training VQ-VAE for %d epochs ===", epochs)

    config = load_config("configs/vqvae.yaml")
    strategy = VQVAEStrategy()
    model = strategy.build_model(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    ds = SpriteDataset(DATA_DIR, image_size=16, transform=PixelArtAugment(flip_prob=0.5))
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            loss_dict = strategy.train_step(model, optimizer, batch)
        if epoch % 100 == 0 or epoch == epochs - 1:
            logger.info("[vqvae] epoch %d: %s", epoch, loss_dict)

    logger.info("=== Stage 1 complete ===")
    return model, strategy, config


def encode_dataset(model: torch.nn.Module) -> torch.Tensor:
    """Encode full dataset through frozen VQ-VAE → (N, 16) index sequences."""
    logger.info("Encoding dataset through frozen VQ-VAE...")
    model.eval()
    raw = load_sprite_tensor(DATA_DIR, image_size=16).to(DEVICE)

    all_indices = []
    with torch.no_grad():
        for i in range(0, len(raw), 32):
            batch = raw[i : i + 32]
            _, indices, _, _ = model.encode(batch)
            # indices: (N, 4, 4) → (N, 16)
            all_indices.append(indices.view(indices.shape[0], -1))

    all_indices = torch.cat(all_indices, dim=0)
    logger.info("Encoded %d images → index sequences of shape %s", len(all_indices), all_indices.shape)
    return all_indices


def stage2_train_prior(
    all_indices: torch.Tensor,
    codebook_size: int,
    epochs: int = 300,
) -> IndexPrior:
    """Train autoregressive prior over codebook index sequences."""
    logger.info("=== Stage 2: Training prior for %d epochs ===", epochs)

    seq_len = all_indices.shape[1]  # 16 for 4x4 grid
    prior = IndexPrior(
        codebook_size=codebook_size,
        seq_len=seq_len,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
    ).to(DEVICE)

    params = sum(p.numel() for p in prior.parameters())
    logger.info("Prior: %d parameters, seq_len=%d, codebook_size=%d", params, seq_len, codebook_size)

    optimizer = torch.optim.Adam(prior.parameters(), lr=3e-4)
    dataset = TensorDataset(all_indices)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for (batch_indices,) in loader:
            batch_indices = batch_indices.to(DEVICE)
            optimizer.zero_grad()

            # Teacher forcing: input is indices[:-1], target is indices[1:]
            # But we predict at all positions: input[i] predicts target[i] = input[i+1]
            logits = prior(batch_indices)  # (N, seq_len, codebook_size)

            # Shift: predict position i+1 from position i
            pred = logits[:, :-1, :].reshape(-1, codebook_size)
            target = batch_indices[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(pred, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 50 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(n_batches, 1)
            logger.info("[prior] epoch %d: ce_loss=%.4f", epoch, avg_loss)

    logger.info("=== Stage 2 complete ===")
    return prior


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_sprite_tensor(DATA_DIR, image_size=16)
    palette = extract_palette(raw, num_colors=2).to(DEVICE)

    # Stage 1: Train VQ-VAE
    model, strategy, config = stage1_train_vqvae(epochs=500)

    # Encode dataset
    all_indices = encode_dataset(model)

    # Stage 2: Train prior
    prior = stage2_train_prior(
        all_indices,
        codebook_size=config.model.codebook_size,
        epochs=300,
    )

    # Sample with prior
    strategy.prior = prior
    model.eval()
    with torch.no_grad():
        samples = strategy.sample(model, 16, DEVICE)
        snapped = snap_to_palette(samples, palette)
    _save_sample_grid(snapped, OUTPUT_DIR / "vqvae_prior_final.png", scale=4)

    # Also sample without prior for comparison
    strategy.prior = None
    with torch.no_grad():
        samples_random = strategy.sample(model, 16, DEVICE)
        snapped_random = snap_to_palette(samples_random, palette)
    _save_sample_grid(snapped_random, OUTPUT_DIR / "vqvae_random_final.png", scale=4)

    logger.info("Done. Results in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""Train conditional diffusion with CFG and compare guidance scales."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.augmentation import PixelArtAugment
from src.data.dataset import SpriteDataset, load_sprite_tensor
from src.palette import extract_palette, snap_to_palette
from src.strategies.diffusion import DiffusionStrategy
from src.trainer import _save_sample_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = Path("data/sprites_colored_focused")
OUTPUT_DIR = Path("data/cfg_comparison")

CLUSTER_NAMES = {
    0: "colorful_objects",
    1: "small_characters",
    2: "cards_ui",
    3: "structures",
    4: "panels_windows",
    5: "faces_heads",
    6: "nature_trees",
    7: "terrain_debris",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config("configs/diffusion.yaml")
    strategy = DiffusionStrategy()
    model = strategy.build_model(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    ds = SpriteDataset(DATA_DIR, image_size=16, transform=PixelArtAugment(flip_prob=0.5))
    raw = load_sprite_tensor(DATA_DIR, image_size=16)
    palette = extract_palette(raw, num_colors=config.dataset.palette_size).to(DEVICE)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    logger.info("p_uncond=%.2f, default guidance_scale=%.1f", strategy.p_uncond, strategy.guidance_scale)

    # Train with label dropout
    epochs = 1000
    for epoch in range(epochs):
        for batch in loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            strategy.train_step(model, optimizer, images, labels=labels)
        if epoch % 200 == 0 or epoch == epochs - 1:
            logger.info("epoch %d/%d", epoch, epochs)

    logger.info("Training complete.")
    model.eval()

    # Compare guidance scales for selected classes
    test_classes = [1, 2, 5, 6]  # characters, cards, faces, nature
    scales = [1.0, 3.0, 5.0, 7.0]

    with torch.no_grad():
        for c in test_classes:
            name = CLUSTER_NAMES[c]
            for scale in scales:
                samples = strategy.sample(model, 8, DEVICE, class_label=c, guidance_scale=scale)
                snapped = snap_to_palette(samples, palette)
                _save_sample_grid(snapped, OUTPUT_DIR / f"class{c}_{name}_scale{scale:.0f}.png", scale=4, grid_cols=8)
            logger.info("Class %d (%s): all scales saved", c, name)

        # Also save real examples for reference
        labels_all = torch.load(DATA_DIR / "labels.pt", weights_only=True)
        for c in test_classes:
            indices = (labels_all == c).nonzero(as_tuple=True)[0][:8]
            _save_sample_grid(raw[indices], OUTPUT_DIR / f"class{c}_{CLUSTER_NAMES[c]}_real.png", scale=4, grid_cols=8)

    logger.info("Done. Results in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""Entry point: load config -> pick strategy -> train."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import TensorDataset

from src.config import load_config
from src.data.augmentation import PixelArtAugment
from src.data.dataset import SpriteDataset, load_sprite_tensor, make_dummy_dataset
from src.palette import extract_palette
from src.strategies.dummy import DummyStrategy
from src.strategies.vae import VAEStrategy
from src.strategies.vqvae import VQVAEStrategy
from src.strategies.diffusion import DiffusionStrategy
from src.trainer import Trainer

STRATEGY_REGISTRY: dict[str, type] = {
    "dummy": DummyStrategy,
    "vae": VAEStrategy,
    "vqvae": VQVAEStrategy,
    "diffusion": DiffusionStrategy,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pixel art generative model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--strategy-override",
        type=str,
        default=None,
        help="Override strategy name (useful for testing with dummy)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    strategy_name = args.strategy_override or config.get("strategy", "dummy")

    if strategy_name not in STRATEGY_REGISTRY:
        logger.error("Unknown strategy: %s. Available: %s", strategy_name, list(STRATEGY_REGISTRY.keys()))
        sys.exit(1)

    strategy = STRATEGY_REGISTRY[strategy_name]()
    logger.info("Using strategy: %s", strategy_name)

    # Load real sprites if available, otherwise fall back to dummy data
    data_path = Path(config.dataset.path)
    if data_path.exists() and any(data_path.glob("*.png")):
        augment = PixelArtAugment(flip_prob=0.5)
        dataset = SpriteDataset(
            data_path,
            image_size=config.dataset.image_size,
            transform=augment,
        )
        # Extract palette from un-augmented data (one-time scan)
        raw_tensor = load_sprite_tensor(data_path, image_size=config.dataset.image_size)
        palette = extract_palette(raw_tensor, num_colors=config.dataset.palette_size)
        logger.info("Loaded %d real sprites from %s", len(dataset), data_path)
    else:
        logger.warning("No sprites at %s — using dummy data", data_path)
        dummy_tensor = make_dummy_dataset(n_samples=128, image_size=config.dataset.image_size)
        dataset = TensorDataset(dummy_tensor)
        palette = None

    trainer = Trainer(strategy=strategy, config=config, strategy_name=strategy_name, palette=palette)
    trainer.train(dataset)


if __name__ == "__main__":
    main()

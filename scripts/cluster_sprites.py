"""Cluster sprites into categories using k-means on raw pixel features.

Produces a labels.pt file in the data directory that SpriteDataset can load
to return (image, label) pairs for conditional training.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.cluster import KMeans

from src.data.dataset import load_sprite_tensor
from src.trainer import _save_sample_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cluster_and_save(
    data_dir: str = "data/sprites_colored_focused",
    n_clusters: int = 8,
    seed: int = 42,
) -> None:
    data_path = Path(data_dir)
    raw = load_sprite_tensor(data_path, image_size=16)
    logger.info("Loaded %d sprites from %s", len(raw), data_path)

    # K-means on flattened pixel features
    features = raw.reshape(raw.shape[0], -1).numpy()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(features)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Save labels
    labels_path = data_path / "labels.pt"
    torch.save(labels_tensor, labels_path)
    logger.info("Saved %d labels to %s", len(labels_tensor), labels_path)

    # Report cluster sizes and save preview grids
    preview_dir = data_path / "cluster_previews"
    preview_dir.mkdir(exist_ok=True)

    for c in range(n_clusters):
        indices = np.where(labels == c)[0]
        logger.info("  Cluster %d: %d sprites", c, len(indices))
        samples = raw[indices[:8]]
        _save_sample_grid(samples, preview_dir / f"cluster_{c}.png", scale=4, grid_cols=8)

    logger.info("Cluster previews saved to %s", preview_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sprites_colored_focused")
    parser.add_argument("--n-clusters", type=int, default=8)
    args = parser.parse_args()

    cluster_and_save(data_dir=args.data_dir, n_clusters=args.n_clusters)

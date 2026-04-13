"""Sprite dataset loading."""

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpriteDataset(Dataset):
    """Loads 16x16 pixel art sprites from a directory of PNGs.

    Returns (C, H, W) float tensors in [0, 1]. If a labels.pt file
    exists in the root directory, returns (image, label) tuples instead.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 16,
        transform: callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.paths = sorted(self.root.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No PNGs found in {self.root}")

        # Load labels if available
        labels_path = self.root / "labels.pt"
        if labels_path.exists():
            self.labels = torch.load(labels_path, weights_only=True)
            self.num_classes = int(self.labels.max().item()) + 1
            logger.info("SpriteDataset: %d sprites, %d classes from %s", len(self.paths), self.num_classes, self.root)
        else:
            self.labels = None
            self.num_classes = 0
            logger.info("SpriteDataset: %d sprites from %s (no labels)", len(self.paths), self.root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGBA")

        # Composite onto black background
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg

        # Nearest-neighbor resize (preserves pixel art crispness)
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.NEAREST)

        # To (C, H, W) float tensor in [0, 1]
        tensor = torch.from_numpy(
            __import__("numpy").array(img, dtype="float32") / 255.0
        ).permute(2, 0, 1)

        if self.transform is not None:
            tensor = self.transform(tensor)

        if self.labels is not None:
            return tensor, self.labels[idx]
        return tensor


def load_sprite_tensor(
    root: str | Path,
    image_size: int = 16,
    transform: callable | None = None,
) -> torch.Tensor:
    """Load all sprites into a single (N, C, H, W) tensor. Convenience for small datasets."""
    dataset = SpriteDataset(root, image_size=image_size, transform=transform)
    images = []
    for i in range(len(dataset)):
        item = dataset[i]
        images.append(item[0] if isinstance(item, tuple) else item)
    return torch.stack(images)


def make_dummy_dataset(
    n_samples: int = 128,
    image_size: int = 32,
    channels: int = 3,
) -> torch.Tensor:
    """Create a dummy dataset of random images for pipeline testing."""
    return torch.rand(n_samples, channels, image_size, image_size)

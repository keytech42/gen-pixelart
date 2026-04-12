"""Color quantization and palette extraction utilities for pixel art."""

import logging

import torch

logger = logging.getLogger(__name__)


def extract_palette(images: torch.Tensor, num_colors: int = 16, max_iters: int = 50) -> torch.Tensor:
    """Extract a palette from a batch of images using k-means clustering.

    Args:
        images: (N, C, H, W) float tensor in [0, 1].
        num_colors: number of palette colors to extract.
        max_iters: k-means iterations.

    Returns:
        (K, C) palette tensor.
    """
    n, c, h, w = images.shape
    pixels = images.permute(0, 2, 3, 1).reshape(-1, c)

    # Subsample for speed on large datasets
    max_pixels = 100_000
    if pixels.shape[0] > max_pixels:
        indices = torch.randperm(pixels.shape[0])[:max_pixels]
        pixels = pixels[indices]

    # K-means
    perm = torch.randperm(pixels.shape[0])[:num_colors]
    centroids = pixels[perm].clone()

    for i in range(max_iters):
        dists = torch.cdist(pixels, centroids)
        assignments = dists.argmin(dim=-1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_colors, device=pixels.device)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, c), pixels)
        counts.scatter_add_(0, assignments, torch.ones(pixels.shape[0], device=pixels.device))

        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)
        # Keep old centroid for empty clusters
        new_centroids[~mask] = centroids[~mask]

        shift = (new_centroids - centroids).norm(dim=-1).max().item()
        centroids = new_centroids
        if shift < 1e-5:
            logger.info("K-means converged at iteration %d", i)
            break

    # Sort by luminance for consistent ordering
    luminance = centroids @ torch.tensor([0.299, 0.587, 0.114], device=centroids.device)
    order = luminance.argsort()
    centroids = centroids[order]

    logger.info("Extracted %d-color palette", num_colors)
    return centroids


def snap_to_palette(images: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Snap continuous pixel values to the nearest palette color.

    Args:
        images: (N, C, H, W) float tensor in [0, 1].
        palette: (K, C) float tensor — the palette colors.

    Returns:
        (N, C, H, W) tensor with each pixel snapped to the nearest palette entry.
    """
    n, c, h, w = images.shape
    pixels = images.permute(0, 2, 3, 1).reshape(-1, c)
    dists = torch.cdist(pixels, palette)
    indices = dists.argmin(dim=-1)
    snapped = palette[indices]
    return snapped.reshape(n, h, w, c).permute(0, 3, 1, 2)


def generate_default_palette(num_colors: int = 16) -> torch.Tensor:
    """Generate a simple evenly-spaced palette for testing."""
    steps = int(round(num_colors ** (1 / 3)))
    colors: list[list[float]] = []
    for r in range(steps):
        for g in range(steps):
            for b in range(steps):
                colors.append([r / max(steps - 1, 1), g / max(steps - 1, 1), b / max(steps - 1, 1)])
                if len(colors) >= num_colors:
                    return torch.tensor(colors)
    return torch.tensor(colors[:num_colors])

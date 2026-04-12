# The Pixel Art Data Pipeline

## What

Loading, preprocessing, and augmenting pixel art sprites for training generative models. This sounds simple — it's images, just load them — but pixel art has constraints that break the standard computer vision pipeline.

## Why here

Pixel art is **not** a natural photograph. Every pixel is deliberate. A pixel artist choosing between placing a pixel at (5, 3) vs. (5, 4) is making an aesthetic decision. This means:

1. **No bilinear/bicubic resizing.** These interpolation methods create new colors that aren't in the palette, turning crisp pixel art into blurry mush. Always use nearest-neighbor (`Image.NEAREST`).

2. **No random crops or color jitter.** A random crop of a 16x16 sprite destroys the composition. Color jitter adds colors outside the palette. These standard augmentations are harmful here.

3. **Palette is a first-class concept.** A 16-color sprite has exactly 16 colors. The model should learn to produce outputs that snap cleanly to a palette, not arbitrary RGB values.

## How

### Dataset (`src/data/dataset.py`)

```python
class SpriteDataset(Dataset):
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGBA")

        # Composite onto black background
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])

        # NEAREST-NEIGHBOR resize only
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.NEAREST)

        # To (C, H, W) float tensor in [0, 1]
        tensor = torch.from_numpy(np.array(img, dtype="float32") / 255.0).permute(2, 0, 1)
```

Key decisions:

**RGBA → RGB with black background.** Sprites often have transparent backgrounds. We composite onto black because: (a) it's a consistent background color, (b) for monochrome sprites, transparent pixels are semantically "empty" which maps naturally to black, (c) it avoids the model having to learn alpha channel semantics.

**Per-access loading, not pre-stacking.** An earlier version loaded all sprites into a single tensor at init time. The problem: augmentations were applied once and frozen. A flip that happened at load time stayed flipped for every epoch. By loading per-`__getitem__`, augmentations are random on every access. This is standard PyTorch DataLoader behavior — the Trainer creates a `DataLoader` that calls `__getitem__` each epoch.

### Palette extraction (`src/palette.py`)

```python
def extract_palette(images, num_colors=16, max_iters=50):
    pixels = images.permute(0, 2, 3, 1).reshape(-1, c)  # All pixels flat
    # Subsample for speed
    if pixels.shape[0] > 100_000:
        pixels = pixels[torch.randperm(pixels.shape[0])[:100_000]]
    # K-means clustering
    centroids = pixels[torch.randperm(pixels.shape[0])[:num_colors]].clone()
    for i in range(max_iters):
        # ... standard k-means update ...
```

K-means over all pixel colors in the dataset. For our monochrome Kenney sprites, this converges in 1-2 iterations to black + white. For colored sprite packs, it finds the dominant palette.

**Why k-means over just reading the palette?** Some sprite formats embed palette metadata, but PNG doesn't always. And when mixing sprites from different sources, the "dataset palette" isn't predefined — it emerges from the data. K-means handles all cases.

**Palette snapping** (`snap_to_palette`) maps each pixel to its nearest palette color by Euclidean distance in RGB space. This is applied to model output before logging — raw float output looks like mush, but palette-snapped output already looks like passable pixel art. The gap between raw and snapped quality is itself informative about model quality.

### Augmentations (`src/data/augmentation.py`)

Only two augmentations are pixel-art-safe:

**Horizontal flip** (`flip_prob=0.5`): Most sprites are roughly symmetric. A flipped sword is still a valid sword. This doubles effective dataset size.

**Palette swap** (`palette_swap_prob`): Find unique colors in a sprite, randomly permute them. For monochrome sprites this is just inversion (black↔white). For colored sprites, a red character becomes blue — still valid pixel art with the same structure. Disabled by default (set to 0.0) because for monochrome training, inverted sprites can confuse the model about which color is "foreground."

```python
def palette_swap(image):
    unique = pixels.unique(dim=0)         # Find all colors used
    perm = torch.randperm(unique.shape[0]) # Random permutation
    shuffled = unique[perm]                # New color mapping
    # Remap each pixel to its shuffled color
```

## Gotchas

**The Kenney 1-Bit Pack is tilesheets, not individual sprites.** The download script (`scripts/download_data.py`) slices the 784x352 packed tilesheet into 1,077 individual 16x16 tiles, skipping fully transparent tiles.

**Palette size must match the data.** `base.yaml` sets `palette_size: 2` for monochrome sprites. Setting it to 16 on 2-color data wastes 14 centroids on near-duplicate blacks. When switching to colored sprites, increase this.

**`drop_last=True` in the DataLoader.** With 1,077 sprites and batch_size=32, the last batch would have 21 sprites. Uneven batch sizes can cause issues with BatchNorm (small batches have noisy statistics). Dropping the last incomplete batch is simpler than handling edge cases.

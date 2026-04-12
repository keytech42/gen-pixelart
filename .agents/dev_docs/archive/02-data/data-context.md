# Task 2: Data — Context

## Key files
- `src/data/dataset.py` — SpriteDataset (PyTorch Dataset) + load_sprite_tensor convenience
- `src/data/augmentation.py` — PixelArtAugment (flip + palette swap)
- `src/palette.py` — extract_palette (k-means), snap_to_palette, generate_default_palette
- `scripts/download_data.py` — downloads Kenney 1-Bit Pack, slices tilesheet into individual PNGs
- `scripts/train.py` — now loads real sprites with dummy fallback
- `configs/base.yaml` — updated to image_size=16, palette_size=2

## Decisions
- **Kenney 1-Bit Pack as starter dataset**: 1,077 monochrome 16x16 sprites, CC0 license. 2-color (black/white) simplifies early training — models converge faster on binary output.
- **Nearest-neighbor resize only**: Bilinear/bicubic creates non-palette colors, breaking pixel art semantics.
- **Transparent pixels composited onto black**: RGBA→RGB via black background. For monochrome sprites this is natural.
- **K-means palette extraction**: Proper implementation even though current dataset is 2-color. Ready for colored sprite packs later.
- **Palette size=2 in base config**: Matches the monochrome dataset. Will increase when switching to colored sprites.
- **Trainer accepts optional palette**: If train.py extracts a palette from real data, it passes it through. Falls back to generate_default_palette.

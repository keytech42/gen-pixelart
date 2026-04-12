# Task 2: Data — Plan

## Goal
Build a sprite dataset pipeline that loads real pixel art, extracts palettes, and applies pixel-art-safe augmentations. End state: Trainer consumes real sprites and logs sample batches to MLflow.

## What we're building
1. **SpriteDataset** (PyTorch Dataset) — loads sprite PNGs from a directory, resizes to target resolution, extracts palette
2. **Palette extraction** — auto-detect dominant colors from a sprite set using k-means
3. **Augmentations** — horizontal flip + palette swap (no random crops or color jitter — these break pixel art semantics)
4. **Data download script** — fetch a small curated sprite set for development

## Design choices
- Dataset yields `(image_tensor, palette)` — label is optional for future conditioning
- Images stored as (C, H, W) float tensors in [0, 1], standard PyTorch convention
- Palette extraction runs once at dataset init, shared across all samples
- Use nearest-neighbor resizing only (no bilinear — that creates non-palette colors)

## Dependencies
- Task 1 (skeleton) — complete
- Need a small sprite dataset (~200-500 sprites) for development

## Done when
- `SpriteDataset` loads real PNGs and returns correct tensor shapes
- Palette extraction produces a valid K-color palette from the dataset
- Augmentations work without breaking pixel art aesthetics
- `train.py` can consume real data (falls back to dummy if no data dir)
- Sample batch logged to MLflow shows recognizable sprites

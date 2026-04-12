# Task 2: Data — Tasks

- [x] SpriteDataset class (PyTorch Dataset, loads PNGs, nearest-neighbor resize)
- [x] Palette extraction via k-means clustering
- [x] Augmentations: horizontal flip, palette swap
- [x] Data download script — slices Kenney 1-Bit Pack tilesheets into 1,077 individual 16x16 sprites
- [x] Integrate SpriteDataset into train.py (with dummy fallback)
- [x] Verify end-to-end: 1,077 real sprites through Trainer on MPS, sample grids logged to MLflow

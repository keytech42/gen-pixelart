# Task 7: Conditioning — Context

## Key files to modify
- `src/data/dataset.py` — SpriteDataset label support
- `src/strategies/base.py` — ABC optional labels
- `src/strategies/diffusion.py` — conditional train_step + sample
- `src/models/unet.py` — class embedding
- `src/trainer.py` — label extraction from batches
- `configs/diffusion.yaml` — num_classes
- `scripts/smoke_test.py` — test conditional path

## New files
- `scripts/cluster_sprites.py` — clustering + label saving
- `scripts/train_conditional.py` — conditional training + per-class sampling
- `docs/learn/05-conditioning.md` — tutorial

## Cluster mapping (from earlier exploration)
- 0: Colorful objects (52)
- 1: Small characters (63)
- 2: Cards/UI (56)
- 3: Wide structures (75)
- 4: Panels/windows (79)
- 5: Faces/heads (125)
- 6: Nature/trees (220)
- 7: Terrain/debris (203)

# Task 7: Conditioning — Tasks

## Phase 1: Labeling
- [x] Cluster sprites and save labels.pt (8 classes, k-means on raw pixels)
- [x] Update SpriteDataset to return (image, label) when labels available

## Phase 2: Conditioning pipeline
- [x] Update GenerativeStrategy ABC with optional labels params
- [x] Update all existing strategies to accept and ignore labels
- [x] Add class embedding to U-Net (num_classes config, added to time embedding)
- [x] Update DiffusionStrategy for class-conditional training and sampling
- [x] Update Trainer to extract labels from batch and pass to strategy

## Phase 3: Training and verification
- [x] Update configs for conditional training (num_classes: 8)
- [x] Smoke test passes (all strategies, conditional and unconditional paths)
- [x] Train conditional diffusion 1000 epochs on labeled colored sprites
- [x] Per-class sampling — cards/characters/structures clearly differentiated

## Docs
- [x] Write 05-conditioning.md tutorial (clustering rationale + conditioning mechanics)

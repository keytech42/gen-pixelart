# Task 7: Class-Conditional Generation — Plan

## Goal
Generate specific types of sprites on demand: "give me a tree", "give me a character." Requires auto-labeling (clustering) + model conditioning.

## What we're building

### Phase 1: Labeling infrastructure
- K-means clustering on raw pixel features → 8 auto-categories
- Save labels alongside sprites, SpriteDataset returns (image, label) pairs
- Educational doc explaining why k-means, alternatives, interpretation

### Phase 2: Conditioning pipeline
- Update GenerativeStrategy ABC with optional `labels` parameter
- Update Trainer to extract and pass labels from DataLoader batches
- Add class embedding to U-Net (added to time embedding path)
- DiffusionStrategy uses class labels during training and sampling
- Conditional sampling: specify class → get that type of sprite

### Phase 3: Training and verification
- Train conditional diffusion on labeled colored sprites
- Sample each class independently, verify visual coherence
- Compare conditional vs unconditional quality

## Key design decisions
- **Conditioning enters through time embedding**: class embed + time embed share the same injection path into ResBlocks. Elegant — ResBlocks don't change.
- **Optional everywhere**: labels=None throughout. Existing unconditional code keeps working.
- **No classifier-free guidance yet**: simple conditioning first. CFG is a natural follow-up.
- **K-means on raw pixels**: 16x16 sprites are small enough that raw pixel clustering works. No need for learned features.

## Done when
- Can sample "class 6" and consistently get trees/nature sprites
- Can sample "class 1" and consistently get character sprites
- Tutorial doc explains the full pipeline from clustering to conditional generation

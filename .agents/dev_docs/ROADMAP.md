# Roadmap

## Completed

| # | Task | What we learned |
|---|------|-----------------|
| 1 | Skeleton | Strategy pattern cleanly separates generative approaches |
| 2 | Data pipeline | Pixel art needs nearest-neighbor everything; augmentation matters |
| 3 | VAE | Reparameterization trick; blurry output from MSE averaging; BCE helps for binary |
| 4 | VQ-VAE | Straight-through estimator; codebook collapse; need a prior for coherent sampling |
| 5 | Diffusion | DDPM/DDIM; noise prediction; iterative refinement produces best quality |
| 6 | Refinements | BCE loss, autoregressive prior, DDIM 16x speedup, colored sprites |
| 7 | Conditioning | K-means clustering; class embedding; conditioning quality ∝ 1/intra-class-variance |

## What's next — prioritized by learning value

### Tier 1: High impact, builds directly on current work

**8. Classifier-Free Guidance (CFG)**
- During training, randomly drop class labels (10-20% of the time)
- At sampling: `eps = eps_uncond + scale * (eps_cond - eps_uncond)`
- Amplifies conditioning signal without retraining from scratch
- Learning value: understand guidance as interpolation in noise-prediction space
- Scope: ~2 hours. Modify DiffusionStrategy train_step + sample, add guidance_scale config

**9. Interactive demo (Gradio)**
- Web UI: select class, click generate, see sprite
- Slider for guidance scale, temperature, sampling steps
- Makes the project tangible and shareable
- Learning value: model serving, inference optimization, UI/UX for ML
- Scope: ~3 hours. New script, Gradio dependency

**10. Higher resolution (32×32)**
- Current: 16×16. Double to 32×32 opens up more detail
- Needs attention layers in U-Net (receptive field matters at larger spatial dims)
- May need larger dataset or data augmentation
- Learning value: understand resolution scaling, attention mechanisms, compute tradeoffs
- Scope: ~4 hours. U-Net attention, new configs, retrain

### Tier 2: Interesting extensions

**11. Latent space exploration**
- VAE: interpolate between two latent vectors → morph between sprites
- Diffusion: DDIM is deterministic → same seed with different classes shows what conditioning changes
- VQ-VAE: swap codebook indices at specific positions → localized edits
- Learning value: understand what latent representations capture
- Scope: ~2 hours. New visualization scripts

**12. Palette conditioning**
- Condition on a target palette (e.g., "generate in NES palette" vs "generate in Game Boy palette")
- Model receives palette as additional input
- Pixel-art-specific — not covered in standard ML literature
- Learning value: multi-signal conditioning, domain-specific design
- Scope: ~4 hours. Palette embedding, data pipeline changes

**13. Evaluation metrics**
- FID (Fréchet Inception Distance) adapted for tiny sprites
- Codebook utilization tracking over training
- Palette adherence score (% of pixels that snap to palette without change)
- Learning value: understand ML evaluation beyond "looks good"
- Scope: ~3 hours. Metrics module, integration with MLflow

### Tier 3: Ambitious

**14. Sprite animation**
- Generate sequences of frames (walk cycle, attack animation)
- Temporal consistency across frames
- Could use video diffusion or autoregressive frame generation
- Learning value: sequential generation, temporal coherence
- Scope: ~8+ hours. New models, new data pipeline, new evaluation

**15. Text conditioning**
- "A knight with a sword" → sprite
- Requires a text encoder (small CLIP or learned from scratch)
- Much larger scope — need text-sprite paired data
- Learning value: multi-modal conditioning, CLIP integration
- Scope: ~10+ hours. Text encoder, paired dataset, cross-attention

## Recommended next steps

**If optimizing for learning**: 8 (CFG) → 11 (latent exploration) → 10 (higher resolution)
CFG is the most important technique you're missing. Latent exploration builds intuition for what models capture. Higher resolution introduces attention and scaling challenges.

**If optimizing for a portfolio piece**: 8 (CFG) → 9 (interactive demo) → 12 (palette conditioning)
CFG makes outputs noticeably better. A Gradio demo is instantly shareable. Palette conditioning is a unique, pixel-art-specific feature that differentiates from generic image generation projects.

**If optimizing for ambition**: 8 (CFG) → 10 (higher resolution) → 14 (animation)
Each step is significantly harder than the last. Animation generation for pixel art is genuinely novel territory.

All paths start with CFG — it's the highest-value next step regardless of direction.

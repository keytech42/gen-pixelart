# Task 8: Classifier-Free Guidance (CFG) — Plan

## Goal
Amplify conditioning signal during sampling without retraining from scratch. CFG interpolates between conditional and unconditional noise predictions, letting you control how strongly the class influences output.

## What we're building
1. **Training change**: Randomly drop class labels (replace with null) during training with probability `p_uncond` (default 10%)
2. **Sampling change**: At each denoising step, compute both conditional and unconditional predictions, then blend: `eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)`
3. **Config**: `guidance_scale` and `p_uncond` parameters

## Why it works
By training with occasional label dropout, the model learns both p(noise|x_t, t, class) and p(noise|x_t, t). At sampling time, the guidance formula amplifies the *difference* between conditional and unconditional predictions — the part that's specifically about the class. Higher guidance_scale = stronger class influence.

## Design
- Use a special "null class" index (num_classes) as the unconditional token
- Expand class embedding to num_classes+1 entries
- During training: replace labels with null_class with probability p_uncond
- During sampling: run model twice per step (conditional + unconditional), blend predictions
- guidance_scale=1.0 = standard conditional, >1.0 = amplified, 0.0 = unconditional

## Done when
- Per-class samples at guidance_scale=3.0 are visibly sharper/more class-specific than scale=1.0
- Cards even more card-like, nature more tree-like

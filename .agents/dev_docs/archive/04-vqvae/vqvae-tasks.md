# Task 4: VQ-VAE — Tasks

- [x] VectorQuantizer module (codebook lookup, straight-through estimator)
- [x] VQVAEModel composing ConvEncoder + VectorQuantizer + ConvDecoder
- [x] VQVAEStrategy implementing GenerativeStrategy ABC
- [x] Register VQ-VAE in train.py + smoke_test.py
- [x] Smoke test passes (dummy + vae + vqvae)
- [x] Train on real sprites — recon_loss 0.25→0.08, codebook utilization ~13%. Random-index samples are blocky (expected without a prior).

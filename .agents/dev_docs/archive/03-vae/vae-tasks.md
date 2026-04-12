# Task 3: VAE — Tasks

- [x] Encoder model (conv layers → mu, log_var)
- [x] Decoder model (latent → conv transpose → image)
- [x] VAE nn.Module wrapping encoder + decoder with reparameterize
- [x] VAEStrategy implementing GenerativeStrategy ABC
- [x] Register VAE in train.py + smoke_test.py strategy registries
- [x] Smoke test passes (both dummy + vae)
- [x] Train on real sprites — recon_loss 0.27→0.01, recognizable shapes in MLflow samples

"""Vector quantization codebook for VQ-VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through estimator.

    Maps continuous encoder features to nearest codebook entries.
    Gradients flow from decoder to encoder via straight-through (stop-gradient copy).

    Args:
        num_embeddings: codebook size K.
        embedding_dim: dimension of each codebook vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output.

        Args:
            z_e: (N, D, H, W) continuous encoder features.

        Returns:
            z_q: (N, D, H, W) quantized features (with straight-through gradient).
            indices: (N, H, W) codebook indices.
            codebook_loss: scalar, moves codebook toward encoder outputs.
            commitment_loss: scalar, keeps encoder outputs near codebook entries.
        """
        # (N, D, H, W) → (N, H, W, D)
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_perm.reshape(-1, self.embedding_dim)

        # Nearest neighbor lookup
        dists = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = dists.argmin(dim=-1)
        z_q_flat = self.embedding(indices)

        # Reshape back to spatial
        z_q_perm = z_q_flat.view(z_e_perm.shape)

        # Losses (before straight-through)
        codebook_loss = F.mse_loss(z_q_perm, z_e_perm.detach())
        commitment_loss = F.mse_loss(z_e_perm, z_q_perm.detach())

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q_perm = z_e_perm + (z_q_perm - z_e_perm).detach()

        # (N, H, W, D) → (N, D, H, W)
        z_q = z_q_perm.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])

        return z_q, indices, codebook_loss, commitment_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert codebook indices back to embeddings.

        Args:
            indices: (N, H, W) integer indices.

        Returns:
            (N, D, H, W) codebook embeddings.
        """
        z_q = self.embedding(indices)  # (N, H, W, D)
        return z_q.permute(0, 3, 1, 2).contiguous()

"""Autoregressive prior over VQ-VAE codebook indices.

A small GPT-style transformer that models the joint distribution of
a 4x4 grid of codebook indices (16 tokens per image). Used to sample
coherent index sequences for VQ-VAE generation.
"""

import torch
import torch.nn as nn


class IndexPrior(nn.Module):
    """Tiny autoregressive transformer over codebook index sequences.

    Input: sequence of codebook indices (flattened 4x4 grid = 16 tokens)
    Output: next-token logits over codebook_size classes at each position

    Uses causal masking so position i can only attend to positions < i.
    """

    def __init__(
        self,
        codebook_size: int,
        seq_len: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(codebook_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, codebook_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for each position.

        Args:
            indices: (N, seq_len) integer tensor of codebook indices.

        Returns:
            (N, seq_len, codebook_size) logits.
        """
        n, s = indices.shape
        positions = torch.arange(s, device=indices.device)
        x = self.token_embed(indices) + self.pos_embed(positions)
        x = self.transformer(x, mask=self.causal_mask[:s, :s])
        return self.head(x)

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressively sample index sequences.

        Args:
            n_samples: number of sequences to generate.
            device: target device.
            temperature: sampling temperature (lower = more deterministic).

        Returns:
            (n_samples, seq_len) integer tensor of codebook indices.
        """
        # Start with a random first token
        indices = torch.randint(0, self.codebook_size, (n_samples, 1), device=device)

        for i in range(1, self.seq_len):
            logits = self.forward(indices)  # (N, current_len, codebook_size)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            indices = torch.cat([indices, next_token], dim=1)

        return indices

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 1024,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even.")

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(positions, inv_freq)

        # Shape: [max_seq_len, head_dim / 2]
        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # q, k:
        # [batch, heads, seq_len, head_dim]

        seq_len = q.size(2)

        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        # Split even and odd dimensions.
        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        q_rotated = torch.stack(
            [
                q_even * cos - q_odd * sin,
                q_even * sin + q_odd * cos,
            ],
            dim=-1,
        ).flatten(-2)

        k_rotated = torch.stack(
            [
                k_even * cos - k_odd * sin,
                k_even * sin + k_odd * cos,
            ],
            dim=-1,
        ).flatten(-2)

        return q_rotated, k_rotated
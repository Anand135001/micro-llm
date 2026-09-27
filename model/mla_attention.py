import torch
import torch.nn as nn
import torch.nn.functional as F


class MLAAttention(nn.Module):
    """
    Factorized Multi-Head Latent Attention (MLA-style).

    KV projections are compressed to a low-rank latent space
    per attention head, then reconstructed before attention.

    RoPE is applied to the reconstructed Q/K tensors.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        latent_rank: int = 24,
        max_seq_len: int = 1024,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.latent_rank = latent_rank
        self.max_seq_len = max_seq_len

        # Query projection
        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # Compress input independently for each head
        self.k_down = nn.Linear(
            d_model,
            num_heads * latent_rank,
            bias=False,
        )

        self.v_down = nn.Linear(
            d_model,
            num_heads * latent_rank,
            bias=False,
        )

        # Per-head reconstruction matrices
        self.k_up = nn.Parameter(
            torch.empty(
                num_heads,
                latent_rank,
                self.head_dim,
            )
        )

        self.v_up = nn.Parameter(
            torch.empty(
                num_heads,
                latent_rank,
                self.head_dim,
            )
        )

        # Output projection
        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # RoPE frequencies
        inv_freq = 1.0 / (
            rope_base
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02,)

        nn.init.normal_(self.k_down.weight, mean=0.0, std=0.02,)

        nn.init.normal_(self.v_down.weight, mean=0.0, std=0.02,)

        nn.init.normal_(self.k_up, mean=0.0, std=0.02,)

        nn.init.normal_(self.v_up, mean=0.0, std=0.02,)

        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02,)

    def _apply_rope(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, H, S, D]
        """

        seq_len = x.size(2)

        positions = torch.arange(seq_len, device=x.device,)

        freqs = torch.outer(positions, self.inv_freq.to(x.device),)

        cos = freqs.cos()[None, None, :, : ]

        sin = freqs.sin()[None, None, :, : ]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        x_rotated = torch.stack(
            (
                rotated_even,
                rotated_odd,
            ),
            dim=-1,
        )

        return x_rotated.flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        # -------------------------
        # Query
        # -------------------------
        q = self.q_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        q = q.transpose(1, 2)

        # -------------------------
        # Latent K
        # -------------------------
        k_latent = self.k_down(x)

        k_latent = k_latent.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.latent_rank,
        )

        # [B, S, H, R] × [H, R, D]
        k = torch.einsum(
            "bshr,hrd->bshd",
            k_latent,
            self.k_up,
        )

        k = k.transpose(1, 2)


        # -------------------------
        # Latent V
        # -------------------------
        v_latent = self.v_down(x)

        v_latent = v_latent.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.latent_rank,
        )

        v = torch.einsum(
            "bshr,hrd->bshd",
            v_latent,
            self.v_up,
        )
        
        v = v.transpose(1, 2)


        # -------------------------
        # RoPE
        # -------------------------
        q = self._apply_rope(q)
        k = self._apply_rope(k)


        # -------------------------
        # Causal attention
        # -------------------------
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,)

        # [B, H, S, D] → [B, S, D]
        out = out.transpose(1, 2).contiguous()

        out = out.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        return self.out_proj(out)
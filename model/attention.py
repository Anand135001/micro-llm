import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
    ) -> None:
        super().__init__()

        if d_model % num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads.")

        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads."
            )

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads

        self.q_proj = nn.Linear(
            d_model,
            num_q_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.num_q_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Repeat K/V heads so that:
        # 2 KV heads → 8 Q heads
        repeat_factor = self.num_q_heads // self.num_kv_heads

        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.view(
            B,
            T,
            self.d_model,
        )

        return self.o_proj(attention_output)
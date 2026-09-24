import torch
import torch.nn as nn

from model.norm import RMSNorm
from model.attention import GQA
from model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        d_ff: int = 1664,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(d_model)

        self.attention = GQA(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
        )

        self.norm2 = RMSNorm(d_model)

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer
        x = x + self.attention(self.norm1(x))

        # Feed-forward sub-layer
        x = x + self.ffn(self.norm2(x))

        return x
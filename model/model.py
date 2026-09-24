import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_block import TransformerBlock
from model.norm import RMSNorm


class MicroLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 512,
        num_layers: int = 12,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        d_ff: int = 1664,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model,)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids:
            [batch, sequence_length]

        returns:
            [batch, sequence_length, vocab_size]
        """

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Weight tying:
        # output projection uses the same matrix
        # as the input embedding.
        logits = F.linear(x, self.token_embedding.weight,)

        return logits
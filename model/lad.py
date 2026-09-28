import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock


class LADCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_iterations: int = 4,
    ):
        super().__init__()

        self.num_iterations = num_iterations

        self.shared_block = TransformerBlock(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            attention_type="gqa",
        )

        # Learned signal telling the shared block which
        # recursion/pass it is currently executing.
        self.iteration_embedding = nn.Parameter(
            torch.zeros(num_iterations, d_model)
        )

        nn.init.normal_(
            self.iteration_embedding,
            mean=0.0,
            std=0.02,
        )

    def forward(self, x):
        for i in range(self.num_iterations):
            x = x + self.iteration_embedding[i]
            x = self.shared_block(x)

        return x
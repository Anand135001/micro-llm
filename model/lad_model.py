import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_block import TransformerBlock
from model.lad import LADCore
from model.norm import RMSNorm


class LADMicroLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 384,
        num_unique_layers: int = 23,
        num_q_heads: int = 6,
        num_kv_heads: int = 2,
        d_ff: int = 1024,
        max_seq_len: int = 1024,
        num_recursions: int = 4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        # First 23 independent layers.
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                attention_type="gqa",
            )
            for _ in range(num_unique_layers)
        ])

        # Shared recurrent section.
        self.lad_core = LADCore(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            num_iterations=num_recursions,
        )

        self.final_norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.lad_core(x)

        x = self.final_norm(x)

        logits = F.linear(
            x,
            self.token_embedding.weight,
        )

        return logits
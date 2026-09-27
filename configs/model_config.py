from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 384
    num_layers: int = 27
    num_q_heads: int = 6
    num_kv_heads: int = 2
    d_ff: int = 1024
    max_seq_len: int = 1024
from dataclasses import dataclass


@dataclass
class GPUTrainConfig:
    # Data
    batch_size: int = 8
    seq_len: int = 1023

    # Gradient accumulation
    gradient_accumulation_steps: int = 8

    # Training budget
    max_training_tokens: int = 100_000_000

    # Validation
    eval_every: int = 250

    # Checkpointing
    save_every: int = 500

    # Logging
    log_every: int = 10
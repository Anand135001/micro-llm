from dataclasses import dataclass


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100_000
    min_learning_rate: float = 3e-5
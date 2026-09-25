import math

import torch

from configs.train_config import TrainConfig


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
):
    def lr_lambda(step: int):
        # Warmup
        if step < config.warmup_steps:
            return float(step + 1) / config.warmup_steps

        # Cosine decay
        progress = (
            step - config.warmup_steps
        ) / (
            config.max_steps - config.warmup_steps
        )

        progress = min(max(progress, 0.0), 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Convert to:
        # min_lr ... max_lr
        min_lr_ratio = (
            config.min_learning_rate
            / config.learning_rate
        )

        return (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    return scheduler
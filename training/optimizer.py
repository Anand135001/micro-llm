import torch

from configs.train_config import TrainConfig


def create_optimizer(model, config: TrainConfig):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    return optimizer

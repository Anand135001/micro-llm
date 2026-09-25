import torch

from model.model import MicroLLM
from configs.train_config import TrainConfig
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler


def main():
    model = MicroLLM()

    config = TrainConfig(
        warmup_steps=5,
        max_steps=20,
        min_learning_rate=3e-5,
    )

    optimizer = create_optimizer(model, config)

    scheduler = create_scheduler(
        optimizer,
        config,
    )

    print("Learning-rate schedule:\n")

    for step in range(21):
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"step={step:2d} "
            f"lr={lr:.8f}"
        )

        optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    main()
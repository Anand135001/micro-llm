import os
import torch

from model.model import MicroLLM
from training.optimizer import create_optimizer
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from configs.train_config import TrainConfig


def main():
    device = torch.device("cpu")

    model = MicroLLM().to(device)

    config = TrainConfig()
    optimizer = create_optimizer(model, config)

    path = "checkpoints/test_checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=123,
        path=path,
    )

    # Create a fresh model + optimizer
    new_model = MicroLLM().to(device)
    new_optimizer = create_optimizer(
        new_model,
        config,
    )

    step = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        path=path,
        device=device,
    )

    print("Loaded step:", step)
    print("Checkpoint test: OK")

    os.remove(path)


if __name__ == "__main__":
    main()
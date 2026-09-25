import torch

from model.model import MicroLLM
from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.trainer import train_step
from configs.train_config import TrainConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    model = MicroLLM().to(device)
    config = TrainConfig()
    optimizer = create_optimizer(model, config)

    loader = create_dataloader(
        batch_size=2,
        validation=False,
    )

    batch = next(iter(loader))

    loss, grad_norm = train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        device=device,
        max_grad_norm=config.max_grad_norm,
    )

    print("Loss:", loss)
    print("Gradient norm:", grad_norm)
    print("Train step: OK")


if __name__ == "__main__":
    main()
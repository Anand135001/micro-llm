import torch

from model.model import MicroLLM
from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.loop import train
from configs.train_config import TrainConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    model = MicroLLM().to(device)
    config = TrainConfig()
    optimizer = create_optimizer(model, config)

    train_loader = create_dataloader(
        batch_size=2,
        validation=False,
    )

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        max_steps=3,
        log_every=1,
        save_every=2,
        checkpoint_dir="checkpoints",
        max_grad_norm=config.max_grad_norm,
    )

    print("Training loop: OK")


if __name__ == "__main__":
    main()
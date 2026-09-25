import torch

from model.model import MicroLLM
from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.amp import create_amp_config
from training.trainer import train_step
from configs.train_config import TrainConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    model = MicroLLM().to(device)
    config = TrainConfig()
    optimizer = create_optimizer(model, config)

    amp_config = create_amp_config(device)

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
        amp_enabled=amp_config["enabled"],
        amp_dtype=amp_config["dtype"],
        scaler=amp_config["scaler"],
    )

    print("Loss:", loss)
    print("Gradient norm:", grad_norm)
    print("AMP train step: OK")


if __name__ == "__main__":
    main()
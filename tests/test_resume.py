import torch

from model.model import MicroLLM
from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.loop import train
from training.checkpoint import load_checkpoint
from configs.train_config import TrainConfig


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = TrainConfig()

    # -------------------------
    # 1. Create fresh model
    # -------------------------
    model = MicroLLM().to(device)
    optimizer = create_optimizer(model, config)

    train_loader = create_dataloader(
        batch_size=2,
        validation=False,
    )

    # -------------------------
    # 2. Train until step 2
    # -------------------------
    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        max_steps=2,
        log_every=1,
        save_every=2,
        checkpoint_dir="checkpoints",
        max_grad_norm=config.max_grad_norm,
    )

    # -------------------------
    # 3. Create NEW model
    # -------------------------
    new_model = MicroLLM().to(device)
    new_optimizer = create_optimizer(
        new_model,
        config,
    )

    # -------------------------
    # 4. Load checkpoint
    # -------------------------
    start_step = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        path="checkpoints/step_2.pt",
        device=device,
    )

    print("Start step:", start_step)

    # -------------------------
    # 5. Resume training
    # -------------------------
    train_loader = create_dataloader(
        batch_size=2,
        validation=False,
    )

    train(
        model=new_model,
        optimizer=new_optimizer,
        train_loader=train_loader,
        device=device,
        max_steps=3,
        start_step=start_step,
        log_every=1,
        save_every=100,
        checkpoint_dir="checkpoints",
        max_grad_norm=config.max_grad_norm,
    )

    print("Resume test: OK")


if __name__ == "__main__":
    main()
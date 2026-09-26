import torch

from model.model import MicroLLM
from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.loop import train
from training.checkpoint import load_checkpoint
from configs.train_config import TrainConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TrainConfig(
        warmup_steps=5,
        max_steps=20,
    )

    # ====== First training run =======
    model = MicroLLM().to(device)

    optimizer = create_optimizer(
        model,
        config,
    )

    scheduler = create_scheduler(
        optimizer,
        config,
    )

    train_loader = create_dataloader(
        batch_size=2,
        validation=False,
    )

    val_loader = create_dataloader(
        batch_size=2,
        validation=True,
    )

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_steps=2,
        gradient_accumulation_steps=2,
        log_every=1,
        eval_every=2,
        save_every=2,
        checkpoint_dir="checkpoints",
        max_grad_norm=config.max_grad_norm,
    )

    # -------------------------
    # New model/optimizer/
    # scheduler
    # -------------------------
    new_model = MicroLLM().to(device)

    new_optimizer = create_optimizer(
        new_model,
        config,
    )

    new_scheduler = create_scheduler(
        new_optimizer,
        config,
    )

    # ====== Load checkpoint ======
    start_step, best_val_loss = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        path="checkpoints/step_2.pt",
        device=device,
    )

    print("Start step:", start_step)
    print("Best validation loss:", best_val_loss)

    # ====== Resume ======
    train(
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_steps=4,
        start_step=start_step,
        gradient_accumulation_steps=2,
        log_every=1,
        eval_every=2,
        save_every=100,
        checkpoint_dir="checkpoints",
        best_val_loss=best_val_loss,
        max_grad_norm=config.max_grad_norm,
    )

    print("Resume test: OK")


if __name__ == "__main__":
    main()
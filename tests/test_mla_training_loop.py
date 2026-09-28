import torch

from model.model import MicroLLM
from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.loop import train
from configs.train_config import TrainConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------
    # MLA model
    # -------------------------
    model = MicroLLM(
        attention_type="mla",
        mla_rank=18,
    ).to(device)

    config = TrainConfig(warmup_steps=5, max_steps=20,)

    optimizer = create_optimizer(model,config,)

    scheduler = create_scheduler(optimizer, config,)

    train_loader = create_dataloader(batch_size=2, validation=False,)

    val_loader = create_dataloader(batch_size=2, validation=True,)

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_steps=4,
        gradient_accumulation_steps=2,
        log_every=1,
        eval_every=2,
        save_every=4,
        checkpoint_dir="checkpoints_mla",
        max_grad_norm=config.max_grad_norm,
    )

    print("MLA training loop: OK")


if __name__ == "__main__":
    main()
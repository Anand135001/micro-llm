import argparse
from pathlib import Path

import torch

from configs.train_config import TrainConfig
from model.model import MicroLLM

from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.loop import train
from training.token_budget import max_optimizer_steps
from training.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train MicroLLM")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-training-tokens",
        type=int,
        default=100_000_000,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_gqa",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
    )
     
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("================================")
    print("MicroLLM Training")
    print("================================")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------
    # Model
    # --------------------------------

    model = MicroLLM(attention_type="gqa").to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------
    # Training configuration
    # --------------------------------

    config = TrainConfig()

    optimizer = create_optimizer(
        model,
        config,
    )

    scheduler = create_scheduler(
        optimizer,
        config,
    )

    # --------------------------------
    # Data
    # --------------------------------

    train_loader = create_dataloader(
        batch_size=args.batch_size,
        validation=False,
    )

    val_loader = create_dataloader(
        batch_size=args.batch_size,
        validation=True,
    )

    # --------------------------------
    # Token budget
    # --------------------------------

    seq_len = 1023

    steps = max_optimizer_steps(
        max_training_tokens=args.max_training_tokens,
        batch_size=args.batch_size,
        seq_len=seq_len,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
    )

    tokens_per_step = (
        args.batch_size
        * seq_len
        * args.gradient_accumulation_steps
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps,
    )

    print(
        "Tokens / optimizer step:",
        f"{tokens_per_step:,}",
    )

    print(
        "Training token budget:",
        f"{args.max_training_tokens:,}",
    )

    print(
        "Optimizer steps:",
        f"{steps:,}",
    )

    # --------------------------------
    # Resume checkpoint
    # --------------------------------

    start_step = 0
    best_val_loss = float("inf")

    if args.resume_from is not None:
        start_step, best_val_loss, metadata = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=args.resume_from,
            device=device,
        )

        print("Resume metadata:", metadata)


    # --------------------------------
    # Training
    # --------------------------------

    best_val_loss = train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_steps=steps,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        start_step=start_step,
        best_val_loss=best_val_loss,
        max_grad_norm=config.max_grad_norm,
    )

    print()
    print("================================")
    print("Training complete")
    print("Best validation loss:", best_val_loss)
    print("================================")


if __name__ == "__main__":
    main()
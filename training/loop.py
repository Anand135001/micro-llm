import time
from pathlib import Path

import torch

from training.amp import create_amp_config
from training.checkpoint import save_checkpoint
from training.evaluate import evaluate
from training.trainer import train_step


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    max_steps,
    log_every=10,
    eval_every=100,
    save_every=1000,
    checkpoint_dir="checkpoints",
    start_step=0,
    max_grad_norm=1.0,
    best_val_loss=float("inf"),
):
    model.train()

    amp_config = create_amp_config(device)

    data_iter = iter(train_loader)

    for step in range(start_step + 1, max_steps + 1):

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Record LR before this optimizer update.
        current_lr = optimizer.param_groups[0]["lr"]

        start_time = time.time()

        loss, grad_norm = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            max_grad_norm=max_grad_norm,
            amp_enabled=amp_config["enabled"],
            amp_dtype=amp_config["dtype"],
            scaler=amp_config["scaler"],
        )

        # Scheduler changes LR for the NEXT step.
        scheduler.step()

        elapsed = time.time() - start_time

        if step % log_every == 0 or step == start_step + 1:
            tokens = batch["input_ids"].numel()
            tokens_per_sec = tokens / elapsed

            print(
                f"step={step:6d} "
                f"loss={loss:.4f} "
                f"grad_norm={grad_norm:.4f} "
                f"lr={current_lr:.8f} "
                f"tokens/s={tokens_per_sec:.1f}"
            )

        # -------------------------
        # Validation
        # -------------------------
        if step % eval_every == 0:
            val_loss = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                max_batches=2,
            )

            print(
                f"step={step:6d} "
                f"val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_val_loss=best_val_loss,
                    path=Path(checkpoint_dir) / "best.pt",
                )

                print(
                    f"New best validation loss: "
                    f"{best_val_loss:.4f}"
                )

        # -------------------------
        # Regular checkpoint
        # -------------------------
        if step % save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                best_val_loss=best_val_loss,
                path=Path(checkpoint_dir) / f"step_{step}.pt",
            )

    return best_val_loss
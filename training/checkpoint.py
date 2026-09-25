from pathlib import Path
import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    path,
    best_val_loss=float("inf"),
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    device,
):
    checkpoint = torch.load(path, map_location=device,)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint["step"]
    best_val_loss = checkpoint.get(
        "best_val_loss",
        float("inf"),
    )

    print(f"Checkpoint loaded: {path}")
    print(f"Resuming from step: {step}")

    return step, best_val_loss
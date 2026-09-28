import torch

from model.model import MicroLLM
from model.lad_model import LADMicroLLM

from training.dataloader import create_dataloader
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.amp import create_amp_config
from training.trainer import train_micro_step
from training.evaluate import evaluate

from configs.train_config import TrainConfig


class FixedBatchLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def collect_batches(loader, num_batches):
    iterator = iter(loader)
    return [next(iterator) for _ in range(num_batches)]


def run_experiment(
    model,
    train_batches,
    val_batches,
    device,
    max_steps=4,
    accumulation_steps=2,
):
    config = TrainConfig(warmup_steps=5, max_steps=20,)

    optimizer = create_optimizer(model, config,)

    scheduler = create_scheduler(optimizer, config,)

    amp_config = create_amp_config(device)

    batch_index = 0
    losses = []

    for step in range(1, max_steps + 1):

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        for _ in range(accumulation_steps):

            batch = train_batches[batch_index]
            batch_index += 1

            loss = train_micro_step(
                model=model,
                batch=batch,
                device=device,
                accumulation_steps=accumulation_steps,
                amp_enabled=amp_config["enabled"],
                amp_dtype=amp_config["dtype"],
                scaler=amp_config["scaler"],
            )

            total_loss += loss

        if amp_config["scaler"] is not None:
            amp_config["scaler"].unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )

        if amp_config["scaler"] is not None:
            amp_config["scaler"].step(optimizer)
            amp_config["scaler"].update()
        else:
            optimizer.step()

        scheduler.step()

        average_loss = (
            total_loss / accumulation_steps
        )

        losses.append(average_loss)

        print(
            f"step={step} "
            f"loss={average_loss:.4f} "
            f"grad_norm={grad_norm.item():.4f}"
        )

    val_loader = FixedBatchLoader(val_batches)

    val_loss = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        max_batches=len(val_batches),
    )

    return losses, val_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = create_dataloader(batch_size=2, validation=False,)

    val_loader = create_dataloader(batch_size=2, validation=True,)

    max_steps = 4
    accumulation_steps = 2

    train_batches = collect_batches(
        train_loader,
        max_steps * accumulation_steps,
    )

    val_batches = collect_batches(
        val_loader,
        2,
    )

    # -------------------------
    # GQA
    # -------------------------

    print("\n========== GQA ==========\n")

    torch.manual_seed(42)

    gqa_model = MicroLLM(attention_type="gqa",).to(device)

    gqa_losses, gqa_val = run_experiment(
        model=gqa_model,
        train_batches=train_batches,
        val_batches=val_batches,
        device=device,
        max_steps=max_steps,
        accumulation_steps=accumulation_steps,
    )

    # -------------------------
    # LAD
    # -------------------------

    print("\n========== LAD ==========\n")

    torch.manual_seed(42)

    lad_model = LADMicroLLM().to(device)

    lad_losses, lad_val = run_experiment(
        model=lad_model,
        train_batches=train_batches,
        val_batches=val_batches,
        device=device,
        max_steps=max_steps,
        accumulation_steps=accumulation_steps,
    )

    print("\n========== RESULTS ==========\n")

    print("GQA train losses:", gqa_losses)
    print("LAD train losses:", lad_losses)

    print(f"GQA validation loss: {gqa_val:.4f}")
    print(f"LAD validation loss: {lad_val:.4f}")


if __name__ == "__main__":
    main()

    
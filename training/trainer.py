import torch

from training.loss import causal_lm_loss


def train_micro_step(
    model,
    batch,
    device,
    accumulation_steps=1,
    amp_enabled=False,
    amp_dtype=torch.float32,
    scaler=None,
):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)

        # Scale loss so accumulated gradient
        # has the correct magnitude.
        scaled_loss = loss / accumulation_steps

    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    return loss.item()
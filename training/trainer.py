import torch

from training.loss import causal_lm_loss


def train_step(
    model,
    optimizer,
    batch,
    device,
    max_grad_norm=1.0,
    amp_enabled=False,
    amp_dtype=torch.float32,
    scaler=None,
):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)

    # Mixed-precision forward pass
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()

        # Unscale before gradient clipping
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        scaler.step(optimizer)
        scaler.update()

    else:
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        optimizer.step()

    return loss.item(), grad_norm.item()
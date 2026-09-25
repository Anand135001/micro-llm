import torch

from training.loss import causal_lm_loss


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device,
    max_batches=10,
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(val_loader):

        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)

        loss = causal_lm_loss(
            logits,
            labels,
        )

        tokens = labels.numel()

        total_loss += loss.item() * tokens
        total_tokens += tokens

    average_loss = total_loss / total_tokens

    model.train()

    return average_loss
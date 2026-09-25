import torch

from training.loss import causal_lm_loss


def train_step(model, optimizer, batch, device, max_grad_norm=1.0):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)

    loss = causal_lm_loss(logits, labels)

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm
    )

    optimizer.step()

    return loss.item(), grad_norm.item()
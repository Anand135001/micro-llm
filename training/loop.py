import time

import torch


def train(
    model,
    optimizer,
    train_loader,
    device,
    max_steps,
    log_every=10,
    max_grad_norm=1.0,
):
    model.train()

    data_iter = iter(train_loader)

    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        start_time = time.time()

        logits = model(input_ids)

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            labels.reshape(-1),
        )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        optimizer.step()

        elapsed = time.time() - start_time

        if step % log_every == 0 or step == 1:
            tokens = input_ids.numel()
            tokens_per_sec = tokens / elapsed

            print(
                f"step={step:5d} "
                f"loss={loss.item():.4f} "
                f"grad_norm={grad_norm.item():.4f} "
                f"tokens/s={tokens_per_sec:.1f}"
            )
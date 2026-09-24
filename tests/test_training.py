import torch

from model.model import MicroLLM
from training.loss import causal_lm_loss
from configs.train_config import TrainConfig


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cpu")

    model = MicroLLM().to(device)

    config = TrainConfig()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    # Tiny fixed dataset.
    # We intentionally reuse the same examples.
    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(
        0,
        model.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    labels = torch.roll(
        input_ids,
        shifts=-1,
        dims=1,
    )

    model.train()

    print("Starting overfit test...\n")

    for step in range(101):
        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)

        loss = causal_lm_loss(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )

        optimizer.step()

        if step % 10 == 0:
            print(
                f"Step {step:03d} | "
                f"Loss {loss.item():.4f}"
            )


if __name__ == "__main__":
    main()
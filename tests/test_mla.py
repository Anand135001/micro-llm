import torch

from model.model import MicroLLM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MicroLLM(
        attention_type="mla",
        mla_rank=18,
    ).to(device)

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    input_ids = torch.randint(
        0,
        model.vocab_size,
        (2, 32),
        device=device,
    )

    labels = torch.randint(
        0,
        model.vocab_size,
        (2, 32),
        device=device,
    )

    logits = model(input_ids)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        labels.reshape(-1),
    )

    loss.backward()

    print("Input shape :", input_ids.shape)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Backward pass: OK")


if __name__ == "__main__":
    main()
import torch

from model.lad_model import LADMicroLLM


def main():
    model = LADMicroLLM()

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", params)

    input_ids = torch.randint(
        0,
        model.vocab_size,
        (2, 32),
    )

    logits = model(input_ids)

    print("Input shape :", input_ids.shape)
    print("Logits shape:", logits.shape)

    labels = torch.randint(
        0,
        model.vocab_size,
        (2, 32),
    )

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        labels.reshape(-1),
    )

    print("Loss:", loss.item())

    loss.backward()

    print("Backward pass: OK")


if __name__ == "__main__":
    main()
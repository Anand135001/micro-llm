import torch

from model.model import MicroLLM


def main() -> None:
    model = MicroLLM()

    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(
        0,
        model.vocab_size,
        (batch_size, seq_len),
    )

    logits = model(input_ids)

    print("Input shape :", input_ids.shape)
    print("Logits shape:", logits.shape)

    expected_shape = (
        batch_size,
        seq_len,
        model.vocab_size,
    )

    assert logits.shape == expected_shape

    targets = torch.randint(
        0,
        model.vocab_size,
        (batch_size, seq_len),
    )

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
    )

    print("Loss:", loss.item())

    loss.backward()

    print("Backward pass: OK")
    print("Full model test passed.")


if __name__ == "__main__":
    main()
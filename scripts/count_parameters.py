from model.model import MicroLLM


def main() -> None:
    model = MicroLLM()

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {total:,}")
    print(f"Trainable parameters: {total / 1_000_000:.3f}M")

    if total > 50_000_000:
        raise RuntimeError(
            "MODEL EXCEEDS THE 50M PARAMETER LIMIT!"
        )

    print("Parameter limit check: PASSED")


if __name__ == "__main__":
    main()
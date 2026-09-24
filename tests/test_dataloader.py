from training.dataloader import create_dataloader


def main() -> None:
    loader = create_dataloader(
        batch_size=2,
        validation=False,
    )

    batch = next(iter(loader))

    print("Input shape :", batch["input_ids"].shape)
    print("Label shape :", batch["labels"].shape)

    print("\nFirst sample:")
    print(batch["input_ids"][0][:20])


if __name__ == "__main__":
    main()
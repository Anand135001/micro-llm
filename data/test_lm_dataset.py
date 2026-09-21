from stream_lm_dataset import StreamingLMDataset


def main() -> None:
    dataset = StreamingLMDataset(
        validation=False,
    )

    for i, sample in enumerate(dataset):
        print(f"Sample {i + 1}")
        print("Input shape :", sample["input_ids"].shape)
        print("Label shape :", sample["labels"].shape)
        print("First 20 input IDs:")
        print(sample["input_ids"][:20].tolist())
        print("First 20 labels:")
        print(sample["labels"][:20].tolist())
        print()

        if i == 2:
            break


if __name__ == "__main__":
    main()
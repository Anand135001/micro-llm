from datasets import load_dataset


def main() -> None:
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    for i, example in enumerate(dataset):
        text = example["text"]

        print(f"\n--- Example {i + 1} ---")
        print(text[:500])

        if i == 2:
            break


if __name__ == "__main__":
    main()
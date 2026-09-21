from pathlib import Path

from datasets import load_dataset


TARGET_BYTES = 100 * 1024 * 1024  # 100 MB
OUTPUT_FILE = Path("data/tokenizer_corpus.txt")


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    total_bytes = 0
    examples = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()

            if not text:
                continue

            f.write(text)
            f.write("\n\n")

            total_bytes += len(text.encode("utf-8"))
            examples += 1

            if examples % 1000 == 0:
                print(
                    f"Examples: {examples:,} | "
                    f"Collected: {total_bytes / (1024 * 1024):.2f} MB"
                )

            if total_bytes >= TARGET_BYTES:
                break

    print("\nFinished.")
    print(f"Examples: {examples:,}")
    print(f"Size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
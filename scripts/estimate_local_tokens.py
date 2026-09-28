from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


PARQUET_PATH = Path(
    "data/fineweb/data/CC-MAIN-2013-20/"
    "train-00000-of-00014.parquet"
)

TOKENIZER_PATH = Path("tokenizer/tokenizer.json")

SAMPLES_PER_ROW_GROUP = 100
MAX_ROW_GROUPS = 20


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    parquet = pq.ParquetFile(PARQUET_PATH)

    total_rows = parquet.metadata.num_rows
    total_row_groups = parquet.num_row_groups

    print("Parquet rows:", total_rows)
    print("Row groups:", total_row_groups)

    # Sample evenly across the shard.
    num_groups = min(
        total_row_groups,
        MAX_ROW_GROUPS,
    )

    if num_groups == 0:
        raise RuntimeError("Parquet contains no row groups.")

    if num_groups == 1:
        group_indices = [0]
    else:
        group_indices = [
            round(
                i * (total_row_groups - 1)
                / (num_groups - 1)
            )
            for i in range(num_groups)
        ]

    total_sample_rows = 0
    total_sample_tokens = 0
    total_sample_chars = 0

    print("\nSampling row groups...\n")

    for group_idx in group_indices:
        table = parquet.read_row_group(
            group_idx,
            columns=["text"],
        )

        texts = table["text"].to_pylist()

        texts = texts[
            :SAMPLES_PER_ROW_GROUP
        ]

        for text in texts:
            if not isinstance(text, str):
                continue

            encoding = tokenizer.encode(text)

            total_sample_tokens += len(encoding.ids)

            total_sample_chars += len(text)
            total_sample_rows += 1

    if total_sample_rows == 0:
        raise RuntimeError("No valid text samples found.")

    avg_tokens_per_doc = (
        total_sample_tokens
        / total_sample_rows
    )

    avg_chars_per_doc = (
        total_sample_chars
        / total_sample_rows
    )

    chars_per_token = (
        total_sample_chars
        / total_sample_tokens
    )

    estimated_total_tokens = (
        avg_tokens_per_doc
        * total_rows
    )

    print("========== RESULT ==========")
    print(
        f"Sampled documents: "
        f"{total_sample_rows:,}"
    )
    print(
        f"Average chars/document: "
        f"{avg_chars_per_doc:.2f}"
    )
    print(
        f"Average tokens/document: "
        f"{avg_tokens_per_doc:.2f}"
    )
    print(
        f"Characters/token: "
        f"{chars_per_token:.3f}"
    )
    print()
    print(
        f"Estimated tokens in shard: "
        f"{estimated_total_tokens:,.0f}"
    )

    # Approximate number of 1024-token packed examples.
    packed_sequences = (
        estimated_total_tokens // 1024
    )

    print(
        f"Approx. 1024-token sequences: "
        f"{packed_sequences:,.0f}"
    )


if __name__ == "__main__":
    main()
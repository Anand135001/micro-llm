import os
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer
from datasets import load_dataset, DownloadConfig


# -------------------------
# Hugging Face cache
# -------------------------
PROJECT_CACHE = (
    Path(__file__).resolve().parent / ".hf_cache"
)
os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_CACHE),
)
# Give slow/unstable connections more time.
os.environ.setdefault(
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "120",
)
os.environ.setdefault(
    "HF_HUB_ETAG_TIMEOUT",
    "30",
)


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
SEQ_LEN = 1024


FINEWEB_DIR = Path(
    os.environ.get(
        "FINEWEB_DIR",
        Path(__file__).resolve().parent / "fineweb",
    )
)

# Local FineWeb-Edu shards
LOCAL_FINEWEB_DIR = FINEWEB_DIR

# Strict mode: never fall back to Hugging Face streaming
LOCAL_ONLY = True


class StreamingLMDataset(IterableDataset):
    def __init__(
        self,
        split: str = "train",
        validation: bool = False,
    ) -> None:
        super().__init__()

        self.split = split
        self.validation = validation

        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

        eos_id = self.tokenizer.token_to_id("[EOS]")

        if eos_id is None:
            raise RuntimeError("[EOS] token not found.")

        self.eos_id = eos_id

    def _documents(self) -> Iterator[str]:
        download_config = DownloadConfig(
            cache_dir=str(PROJECT_CACHE),
            max_retries=5,
        )

        parquet_files = sorted(
            LOCAL_FINEWEB_DIR.glob(
                "data/CC-MAIN-2013-20/*.parquet"
            )
        )

        if not parquet_files:
            if LOCAL_ONLY:
                raise FileNotFoundError(
                    "No local FineWeb-Edu parquet shards found. "
                    "Download the required shards before training."
                )

            print("No local FineWeb-Edu shards found.")
            print("Using Hugging Face streaming.")

            dataset = load_dataset(
                DATASET_NAME,
                split=self.split,
                streaming=True,
                download_config=download_config,
            )
        else:
            print(
                f"Using {len(parquet_files)} local FineWeb-Edu shard(s)."
            )

            dataset = load_dataset(
                "parquet",
                data_files=[str(path) for path in parquet_files],
                split="train",
                streaming=True,
            )

        for index, example in enumerate(dataset):
            # Deterministic 1% validation split.
            is_validation = (index % 100 == 0)

            if is_validation != self.validation:
                continue

            text = example["text"]

            if text and text.strip():
                yield text

    def __iter__(self):
        buffer = []

        for text in self._documents():
            token_ids = self.tokenizer.encode(text).ids

            # Separate documents.
            token_ids.append(self.eos_id)

            buffer.extend(token_ids)

            while len(buffer) >= SEQ_LEN:
                sequence = buffer[:SEQ_LEN]
                buffer = buffer[SEQ_LEN:]

                input_ids = torch.tensor(
                    sequence,
                    dtype=torch.long,
                )

                yield {
                    "input_ids": input_ids[:-1],
                    "labels": input_ids[1:],
                }
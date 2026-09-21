from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from tokenizers import Tokenizer


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

SEQ_LEN = 1024


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
        dataset = load_dataset(
            DATASET_NAME,
            split=self.split,
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
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


VOCAB_SIZE = 16_384

INPUT_FILE = Path("data/tokenizer_sample.txt")
OUTPUT_DIR = Path("tokenizer")
OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
        ],
    )

    tokenizer.train(
        files=[str(INPUT_FILE)],
        trainer=trainer,
    )

    tokenizer.save(str(OUTPUT_FILE))

    print("Tokenizer trained successfully.")
    print(f"Requested vocabulary size: {VOCAB_SIZE}")
    print(f"Actual vocabulary size:    {tokenizer.get_vocab_size()}")
    print(f"Saved to:                   {OUTPUT_FILE}")

    text = "The Earth revolves around the Sun."
    encoded = tokenizer.encode(text)

    print("\nTest text:")
    print(text)

    print("\nToken IDs:")
    print(encoded.ids)

    print("\nTokens:")
    print(encoded.tokens)

    print("\nDecoded:")
    print(tokenizer.decode(encoded.ids))


if __name__ == "__main__":
    main()
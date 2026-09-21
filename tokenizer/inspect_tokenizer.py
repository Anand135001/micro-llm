from tokenizers import Tokenizer


TOKENIZER_PATH = "tokenizer/tokenizer.json"


def inspect(text: str, tokenizer: Tokenizer) -> None:
    encoded = tokenizer.encode(text)

    print("=" * 60)
    print(f"Text: {text}")
    print(f"Token count: {len(encoded.ids)}")

    print("\nIDs:")
    print(encoded.ids)

    print("\nTokens:")
    print(encoded.tokens)

    print("\nDecoded:")
    print(repr(tokenizer.decode(encoded.ids)))


def main() -> None:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    print("Vocabulary size:", tokenizer.get_vocab_size())

    print("\nSpecial token IDs:")
    for token in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]:
        print(f"{token}: {tokenizer.token_to_id(token)}")

    texts = [
        "Hello world!",
        "The Earth revolves around the Sun.",
        "Python is useful for machine learning.",
        "Transformers use attention.",
        "12345 + 67890 = 80235",
        "hello,HELLO Hello",
    ]

    for text in texts:
        inspect(text, tokenizer)


if __name__ == "__main__":
    main()
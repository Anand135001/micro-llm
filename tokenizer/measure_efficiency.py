from tokenizers import Tokenizer


TOKENIZER_PATH = "tokenizer/tokenizer.json"


def main() -> None:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    text = """
    Transformers are neural networks designed to process sequences.
    They use attention mechanisms to determine which parts of the input
    are relevant to each other.
    """

    encoded = tokenizer.encode(text)

    characters = len(text)
    tokens = len(encoded.ids)

    print(f"Characters: {characters}")
    print(f"Tokens:     {tokens}")
    print(f"Chars/token:{characters / tokens:.2f}")


if __name__ == "__main__":
    main()
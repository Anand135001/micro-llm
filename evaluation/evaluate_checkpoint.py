import math
from pathlib import Path

import torch

from model.model import MicroLLM
from tokenizers import Tokenizer
from training.dataloader import create_dataloader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints_gqa_100m"
    / "best.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "tokenizer.json"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 32)
    print("MicroLLM Evaluation")
    print("=" * 32)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -----------------------------
    # Model
    # -----------------------------

    model = MicroLLM(
        attention_type="gqa"
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Checkpoint step:",
        checkpoint["step"],
    )

    print(
        "Best recorded val loss:",
        checkpoint["best_val_loss"],
    )

    # -----------------------------
    # Validation data
    # -----------------------------

    val_loader = create_dataloader(
        batch_size=8,
        validation=True,
    )

    total_loss = 0.0
    total_tokens = 0

    max_batches = 100

    print()
    print(
        f"Evaluating {max_batches} "
        "validation batches..."
    )

    with torch.inference_mode():

        for batch_index, batch in enumerate(
            val_loader
        ):

            if batch_index >= max_batches:
                break

            input_ids = batch[
                "input_ids"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            logits = model(input_ids)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            token_count = labels.numel()

            total_loss += (
                loss.item() * token_count
            )

            total_tokens += token_count

            if (
                (batch_index + 1) % 10 == 0
                or batch_index == 0
            ):
                current_loss = (
                    total_loss
                    / total_tokens
                )

                print(
                    f"batch={batch_index + 1:3d} "
                    f"loss={current_loss:.4f}"
                )

    final_loss = total_loss / total_tokens
    perplexity = math.exp(final_loss)

    print()
    print("=" * 32)
    print("Evaluation complete")
    print("=" * 32)

    print(
        f"Validation loss: {final_loss:.4f}"
    )

    print(
        f"Perplexity:       {perplexity:.4f}"
    )

    print(
        f"Tokens evaluated:  {total_tokens:,}"
    )


if __name__ == "__main__":
    main()
import torch

from model.model import MicroLLM
from training.dataloader import create_dataloader
from training.evaluate import evaluate


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MicroLLM().to(device)

    val_loader = create_dataloader(
        batch_size=2,
        validation=True,
    )

    val_loss = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
        max_batches=2,
    )

    print("Validation loss:", val_loss)
    print("Evaluation: OK")


if __name__ == "__main__":
    main()
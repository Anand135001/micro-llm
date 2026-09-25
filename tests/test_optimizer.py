import torch

from model.model import MicroLLM
from configs.train_config import TrainConfig
from training.optimizer import create_optimizer


def main():
    model = MicroLLM()

    config = TrainConfig()

    optimizer = create_optimizer(model, config)

    print("Optimizer:", optimizer.__class__.__name__)
    print("Learning rate:", optimizer.param_groups[0]["lr"])
    print("Weight decay:", optimizer.param_groups[0]["weight_decay"])

    # Small sanity check
    input_ids = torch.randint(0, model.vocab_size, (2, 32))

    labels = torch.randint(0, model.vocab_size, (2, 32))

    logits = model(input_ids)

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        labels.reshape(-1)
    )

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print("Loss:", loss.item())
    print("Optimizer step: OK")


if __name__ == "__main__":
    main()
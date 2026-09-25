import torch

from training.amp import create_amp_config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_config = create_amp_config(device)

    print("Device:", device)
    print("AMP enabled:", amp_config["enabled"])
    print("AMP dtype:", amp_config["dtype"])
    print("GradScaler:", amp_config["scaler"])

    print("AMP configuration: OK")


if __name__ == "__main__":
    main()
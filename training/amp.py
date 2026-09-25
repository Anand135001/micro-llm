import torch


def create_amp_config(device):
    if device.type != "cuda":
        return {
            "enabled": False,
            "dtype": torch.float32,
            "scaler": None,
        }

    if torch.cuda.is_bf16_supported():
        return {
            "enabled": True,
            "dtype": torch.bfloat16,
            "scaler": None,
        }

    scaler = torch.amp.GradScaler("cuda")

    return {
        "enabled": True,
        "dtype": torch.float16,
        "scaler": scaler,
    }
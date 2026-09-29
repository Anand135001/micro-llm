import torch


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("No CUDA GPU detected.")
        return

    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        print(f"\nGPU {i}")
        print("Name:", props.name)
        print(
            "VRAM:",
            round(
                props.total_memory / 1024**3,
                2,
            ),
            "GB",
        )

    print(
        "\nCurrent device:",
        torch.cuda.current_device(),
    )


if __name__ == "__main__":
    main()
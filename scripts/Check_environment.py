import platform
import torch  # pyright: ignore[reportMissingImports]

def main() -> None:
    print("====== Environment Check =======")
    print(f"Python platform : {platform.platform()}")
    print(f"Pytorch version : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    z = x @ y

    print(f"Tensor test : {z.shape}")
    print("Environment OK")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Root Mean Square:
        # sqrt(mean(x²))
        rms = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(rms + self.eps)

        return x * self.weight
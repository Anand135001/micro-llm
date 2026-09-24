import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    logits:
        [B, T, V]

    labels:
        [B, T]
    """

    batch_size, seq_len, vocab_size = logits.shape

    return F.cross_entropy(
        logits.reshape(
            batch_size * seq_len,
            vocab_size,
        ),
        labels.reshape(
            batch_size * seq_len,
        ),
    )
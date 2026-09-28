def tokens_per_optimizer_step(
    batch_size: int,
    seq_len: int,
    gradient_accumulation_steps: int,
) -> int:
    return (
        batch_size
        * seq_len
        * gradient_accumulation_steps
    )


def max_optimizer_steps(
    max_training_tokens: int,
    batch_size: int,
    seq_len: int,
    gradient_accumulation_steps: int,
) -> int:
    tokens_per_step = tokens_per_optimizer_step(
        batch_size=batch_size,
        seq_len=seq_len,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    return max_training_tokens // tokens_per_step
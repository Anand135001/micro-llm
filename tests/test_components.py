import torch

from model.norm import RMSNorm
from model.swiglu import SwiGLU
from model.attention import GQA
from model.rope import RotaryEmbedding
from model.transformer_block import TransformerBlock

def main() -> None:
    batch_size = 2
    seq_len = 32
    d_model = 512

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    print("Input shape:", x.shape)

    # -------------------------
    # RMSNorm
    # -------------------------
    norm = RMSNorm(d_model)

    norm_out = norm(x)

    print("RMSNorm output:", norm_out.shape)

    assert norm_out.shape == x.shape


    # -------------------------
    # SwiGLU
    # -------------------------
    ffn = SwiGLU(
        d_model=d_model,
        d_ff=1664,
    )

    ffn_out = ffn(x)

    print("SwiGLU output:", ffn_out.shape)

    assert ffn_out.shape == x.shape


    # -------------------------
    # GQA
    # -------------------------
    attention = GQA(
        d_model=512,
        num_q_heads=8,
        num_kv_heads=2,
    )

    attention_out = attention(x)

    print("GQA output:", attention_out.shape)

    assert attention_out.shape == x.shape


    # ======= RoPE =========
    
    rope = RotaryEmbedding(
        head_dim=64,
        max_seq_len=1024,
    )

    q = torch.randn(2, 8, 32, 64)
    k = torch.randn(2, 2, 32, 64)

    q_rope, k_rope = rope(q, k)

    print("RoPE Q output:", q_rope.shape)
    print("RoPE K output:", k_rope.shape)

    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape


    # ======= Transformer Block ========
    block = TransformerBlock(
        d_model=512,
        num_q_heads=8,
        num_kv_heads=2,
        d_ff=1664,
        max_seq_len=1024,
    )

    block_out = block(x)
    print("Transformer block output:", block_out.shape)
    assert block_out.shape == x.shape


    print("\nAll component tests passed.")

if __name__ == "__main__":
    main()
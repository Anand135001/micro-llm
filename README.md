# MicroLLM

A small decoder-only language model built and trained from scratch under a strict **50M trainable-parameter budget**.

> **Status: Work in Progress**

MicroLLM is an end-to-end learning and research project focused on understanding the complete lifecycle of a language model:

**tokenization → dataset preparation → transformer architecture → training → evaluation → inference → scaling**

The long-term goal is to study how a small language model changes as training compute and model scale increase, beginning with a planned **0 → 1B-token training run** and later moving to larger models.

---

## Overview

MicroLLM is a custom Transformer-based causal language model implemented directly in PyTorch.

The project intentionally avoids starting from a pretrained language model. The architecture, tokenizer, data pipeline, training loop, checkpointing, evaluation infrastructure, and inference API are built as part of the project.

The current primary architecture uses:

- Decoder-only Transformer
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU feed-forward layers
- Tied input/output embeddings
- Byte-Level BPE tokenizer
- 1,024-token context length

The current model contains **48,779,904 trainable parameters**, keeping it below the 50M parameter budget.

---

## Model Configuration

| Configuration | Value |
|---|---:|
| Trainable parameters | **48,779,904 (~48.78M)** |
| Vocabulary size | 16,384 |
| Transformer layers | 27 |
| Hidden size | 384 |
| Query heads | 6 |
| Key/Value heads | 2 |
| Attention | GQA |
| Head dimension | 64 |
| FFN dimension | 1,024 |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Context length | 1,024 |
| Biases | Disabled |
| Embeddings | Tied |

---

## Architecture

```text
Input Text
    │
    ▼
Byte-Level BPE Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Token Embedding
    │
    ▼
27 × Transformer Block
    │
    ├── RMSNorm
    ├── Grouped Query Attention
    │     └── RoPE
    ├── Residual Connection
    ├── RMSNorm
    ├── SwiGLU
    └── Residual Connection
    │
    ▼
Final RMSNorm
    │
    ▼
Tied LM Head
    │
    ▼
Next-Token Logits
```

---

## Why this architecture?

The model is deliberately relatively deep while keeping the hidden dimension modest so that the complete network stays within the parameter budget.

Grouped Query Attention reduces the number of key/value heads while retaining multiple query heads. The architecture also uses RoPE, RMSNorm, and SwiGLU as core Transformer components.

The current configuration is the **primary GQA control model**. Alternative architecture experiments are kept separate from it so that the main training path remains reproducible.

---

## Tokenizer

MicroLLM uses a **Byte-Level BPE tokenizer** with a vocabulary of:

```text
16,384 tokens
```

Special tokens:

```text
[PAD]
[UNK]
[BOS]
[EOS]
```

Tokenizer file:

```text
tokenizer/tokenizer.json
```

The tokenizer was trained on a sample of FineWeb-Edu data.

---

## Dataset

The current training corpus is based on:

**HuggingFaceFW/FineWeb-Edu**

The training pipeline is designed around locally stored parquet shards so training does not depend on continuous network streaming.

```text
FineWeb-Edu
    ↓
Documents
    ↓
Byte-Level BPE tokenizer
    ↓
Token IDs
    ↓
EOS separation
    ↓
Packed 1024-token sequences
    ↓
Next-token prediction
```

A deterministic validation split is used for development and training monitoring.

---

## Training

Training is implemented from scratch in PyTorch.

Current training components include:

- AdamW optimizer
- Gradient accumulation
- Gradient clipping
- Linear warmup
- Cosine learning-rate decay
- Checkpoint saving
- Checkpoint resume
- Validation during training
- Optional AMP support
- Token-throughput logging

Training is controlled primarily by token budget rather than epoch count.

Example:

```bash
python -m scripts.train \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --max-training-tokens 100000000 \
    --eval-every 500 \
    --save-every 2000 \
    --log-every 100
```

---

## Hardware

Development and training experiments have used:

- Local CPU environment for development and testing
- NVIDIA Tesla T4 for GPU training

For the current 48.78M-parameter architecture, a Tesla T4 reached roughly **4.1k tokens/sec** during the observed GPU training runs with batch size 8 and a 1,023-token training sequence.

Actual throughput varies with environment and runtime conditions.

---

## Current Results

A **100M-token** training run was completed successfully using the primary GQA model.

### 100M-token checkpoint

```text
Model parameters:       48,779,904
Training budget:        100M tokens
Checkpoint step:        12,000
Best recorded val loss: 3.8326
```

A separate validation pass over **818,400 tokens** produced:

```text
Validation loss: 3.9628
Perplexity:       52.6029
Tokens evaluated: 818,400
```

The 100M checkpoint is the currently retained trained model artifact.

> A later 250M-token experiment produced a lower validation loss during training, but its checkpoint was not retained. Therefore, the 250M run is not treated as the current final model artifact.

---

## Evaluation

Evaluation infrastructure is being built around `lm-evaluation-harness`.

The initial integration test successfully executed:

- HellaSwag
- ARC-Easy
- PIQA
- WinoGrande

The initial benchmark run used only two samples per task and was intended only as a correctness test. Those results are **not considered final benchmark results**.

Planned final evaluations include:

- HellaSwag
- ARC-Easy
- PIQA
- WinoGrande
- Additional language-model evaluation
- Held-out perplexity measurements

---

## Inference

A lightweight FastAPI backend is being developed to serve the trained model locally.

Current architecture:

```text
React Frontend
      │
      ▼
FastAPI Backend
      │
      ▼
MicroLLM
      │
      ▼
Tokenizer
      │
      ▼
Generated Text
```

Current endpoints include:

```text
GET  /health
POST /generate
POST /generate/stream
```

Streaming generation is being developed for the interactive frontend.

---

## Project Structure

```text
micro-llm/
│
├── backend/
│   └── app.py
│
├── configs/
│   └── train_config.py
│
├── data/
│   └── stream_lm_dataset.py
│
├── evaluation/
│   └── ...
│
├── frontend/
│   └── ...
│
├── model/
│   ├── attention.py
│   ├── mla_attention.py
│   ├── model.py
│   ├── norm.py
│   ├── rope.py
│   ├── swiglu.py
│   └── transformer_block.py
│
├── scripts/
│   ├── train.py
│   ├── count_parameters.py
│   ├── check_gpu.py
│   └── ...
│
├── tests/
│   └── ...
│
├── tokenizer/
│   └── tokenizer.json
│
├── training/
│   ├── checkpoint.py
│   ├── dataloader.py
│   ├── loop.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── ...
│
├── README.md
└── requirements.txt
```

---

## Checkpointing and Resume

Checkpoints store the complete training state needed for continuation:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
step
best_val_loss
metadata
```

This allows training to continue across separate GPU sessions instead of restarting from zero.

Large checkpoint files are kept outside Git rather than committed directly to the repository.

---

## Experiments

The repository also contains experimental implementations for alternative ideas, including:

- MLA-style attention
- Learned/recurrent depth
- Parameter sharing

These experiments are separated from the primary GQA configuration so that the main model remains a reproducible control.

---

## Roadmap

### Phase 1 — Core model

- [x] Build Transformer architecture
- [x] Implement GQA
- [x] Implement RoPE
- [x] Implement RMSNorm
- [x] Implement SwiGLU
- [x] Implement tied embeddings
- [x] Keep model below 50M parameters
- [x] Build tokenizer
- [x] Build training pipeline

### Phase 2 — Training

- [x] GPU training
- [x] Checkpoint saving
- [x] Checkpoint resume
- [x] Validation monitoring
- [x] 100M-token run
- [ ] Clean 0 → 1B-token training run
- [ ] Analyze training scaling behavior

### Phase 3 — Evaluation

- [x] Validation loss
- [x] Perplexity
- [x] Benchmark infrastructure
- [ ] Full HellaSwag evaluation
- [ ] Full ARC-Easy evaluation
- [ ] Full PIQA evaluation
- [ ] Full WinoGrande evaluation
- [ ] Additional evaluation

### Phase 4 — Inference

- [x] FastAPI backend
- [x] Local generation
- [ ] Streaming generation
- [ ] React chat interface
- [ ] Inference performance measurements

### Phase 5 — Scaling

After the 1B-token experiment:

```text
MicroLLM
   ↓
Analyze training behavior
   ↓
Increase model size
   ↓
Train larger models
   ↓
Compare scaling and architecture choices
```

---

## Long-Term Goal

The long-term experiment is to study how the same model changes as training tokens increase:

```text
0
│
├── 100M
├── 250M
├── 500M
└── 1B tokens
        │
        ▼
Analyze loss, perplexity,
benchmarks and generation
        │
        ▼
Move to larger models
```

The intention is to use the project as a practical foundation for learning language-model engineering, training systems, evaluation, and scaling.

---

## Limitations

MicroLLM is still an experimental project.

Current limitations include:

- Small model capacity compared with large modern language models
- Limited training compute
- Limited training data relative to large-scale LLM training
- Benchmark evaluation is not yet complete
- Inference is currently local
- No public production deployment
- The current retained checkpoint represents the 100M-token run

The project prioritizes understanding and experimentation over production-scale performance.

---

## License

License information will be added before the project reaches a stable release.

---

## Status

**Active development.**

The current focus is rebuilding the training experiment cleanly from **0 → 1B tokens**, preserving checkpoints and measurements at each milestone before moving to larger models.

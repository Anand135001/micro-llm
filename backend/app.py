import json
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

from model.model import MicroLLM


# --------------------------------
# Paths
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"

# Change this later if you download the newer checkpoint.
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints_gqa_100m"
    / "best.pt"
)


# --------------------------------
# Device
# --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")

if bos_id is None or eos_id is None:
    raise RuntimeError("[BOS] or [EOS] token not found.")


# --------------------------------
# Model
# --------------------------------

model = MicroLLM(attention_type="gqa").to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Checkpoint loaded:", CHECKPOINT_PATH)

print("Checkpoint step:", checkpoint.get("step"))

print("Best validation loss:", checkpoint.get("best_val_loss"))


# --------------------------------
# FastAPI
# --------------------------------

app = FastAPI(
    title="MicroLLM API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------
# Request / response
# --------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 50


class GenerateResponse(BaseModel):
    text: str


# --------------------------------
# Generation
# --------------------------------

@torch.inference_mode()
def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:

    encoded = tokenizer.encode(prompt)

    input_ids = encoded.ids

    if not input_ids:
        input_ids = [bos_id]

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):

        # Keep within model context.
        input_for_model = input_ids[
            :, -1024:
        ]

        logits = model(
            input_for_model
        )

        next_token_logits = logits[
            :, -1, :
        ]

        # Temperature
        temperature = max(
            temperature,
            1e-5,
        )

        next_token_logits = (
            next_token_logits
            / temperature
        )

        # Top-k
        if top_k > 0:
            k = min(
                top_k,
                next_token_logits.size(-1),
            )

            values, _ = torch.topk(
                next_token_logits,
                k,
            )

            threshold = values[
                :, -1
            ].unsqueeze(-1)

            next_token_logits = torch.where(
                next_token_logits < threshold,
                torch.full_like(
                    next_token_logits,
                    float("-inf"),
                ),
                next_token_logits,
            )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        if next_token.item() == eos_id:
            break

    generated_ids = input_ids[
        0
    ].tolist()

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return text


# --------------------------------
# Health endpoint
# --------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "MicroLLM",
        "parameters": 48_779_904,
        "device": str(device),
        "checkpoint_step": checkpoint.get(
            "step"
        ),
    }


# --------------------------------
# Generate endpoint
# --------------------------------

@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(request: GenerateRequest):

    text = generate_text(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    )

    return GenerateResponse(
        text=text
    )


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest):

    return StreamingResponse(
        stream_generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@torch.inference_mode()
def stream_generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    encoded = tokenizer.encode(prompt)

    input_ids_list = encoded.ids

    if not input_ids_list:
        input_ids_list = [bos_id]

    input_ids = torch.tensor(
        [input_ids_list],
        dtype=torch.long,
        device=device,
    )

    previous_text = tokenizer.decode(
        input_ids_list,
        skip_special_tokens=True,
    )

    for _ in range(max_new_tokens):

        input_for_model = input_ids[:, -1024:]
        logits = model(input_for_model)
        next_token_logits = logits[:, -1, :]

        temperature = max(temperature, 1e-5)

        next_token_logits = (next_token_logits / temperature)

        if top_k > 0:
            k = min(top_k, next_token_logits.size(-1),)

            values, _ = torch.topk(next_token_logits, k,)

            threshold = values[:, -1].unsqueeze(-1)

            next_token_logits = torch.where(
                next_token_logits < threshold,
                torch.full_like(
                    next_token_logits,
                    float("-inf"),
                ),
                next_token_logits,
            )

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        generated_text = tokenizer.decode(
            input_ids[0].tolist(),
            skip_special_tokens=True,
        )

        # Send only newly generated text.
        delta = generated_text[len(previous_text):]

        if delta:
            event = {
                "text": delta,
                "done": False,
            }

            yield (f"data: {json.dumps(event)}\n\n")

            previous_text = generated_text

        if next_token.item() == eos_id:
            break

    yield (
        "data: "
        + json.dumps({
            "text": "",
            "done": True,
        })
        + "\n\n"
    )
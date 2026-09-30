import math
from pathlib import Path
import os

import torch
from lm_eval.api.model import LM
from lm_eval import simple_evaluate
from tokenizers import Tokenizer

from model.model import MicroLLM


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = Path(
    os.environ.get(
        "MICROLLM_CHECKPOINT",
        str(
            PROJECT_ROOT
            / "checkpoints_gqa_100m"
            / "best.pt"
        ),
    )
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "tokenizer.json"
)


class MicroLLMEval(LM):
    def __init__(
        self,
        checkpoint_path,
        batch_size=8,
        device="cuda",
    ):
        super().__init__()

        self._batch_size = batch_size
        self._device = torch.device(device)

        self.model = MicroLLM(
            attention_type="gqa"
        ).to(self._device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.tokenizer = Tokenizer.from_file(
            str(TOKENIZER_PATH)
        )

        self._bos_id = self.tokenizer.token_to_id(
            "[BOS]"
        )

        self._eos_id = self.tokenizer.token_to_id(
            "[EOS]"
        )

        print(
            "Loaded checkpoint:",
            checkpoint_path,
        )

        print(
            "Checkpoint step:",
            checkpoint["step"],
        )

        print(
            "Recorded best val loss:",
            checkpoint["best_val_loss"],
        )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def max_length(self):
        return 1024

    @property
    def max_gen_toks(self):
        return 128

    @property
    def eot_token_id(self):
        return self._eos_id

    def tok_encode(
        self,
        string,
        add_special_tokens=None,
        **kwargs,
    ):
        return self.tokenizer.encode(
            string
        ).ids

    def tok_decode(
        self,
        tokens,
        skip_special_tokens=True,
    ):
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=skip_special_tokens,
        )

    def _encode_pair(
        self,
        context,
        continuation,
    ):
        if context == "":
            continuation_ids = self.tok_encode(
                continuation
            )

            context_ids = []

            if self._bos_id is not None:
                context_ids = [
                    self._bos_id
                ]

            return (
                context_ids,
                continuation_ids,
            )

        # Match causal LM boundary handling.
        n_spaces = (
            len(context)
            - len(context.rstrip())
        )

        if n_spaces > 0:
            continuation = (
                context[-n_spaces:]
                + continuation
            )

            context = context[:-n_spaces]

        whole_ids = self.tok_encode(
            context + continuation
        )

        context_ids = self.tok_encode(
            context
        )

        context_len = len(context_ids)

        continuation_ids = whole_ids[
            context_len:
        ]

        return (
            context_ids,
            continuation_ids,
        )

    @torch.inference_mode()
    def loglikelihood(
        self,
        requests,
        disable_tqdm=False,
    ):
        results = []

        encoded_requests = []

        for request in requests:
            context, continuation = request.args

            context_ids, continuation_ids = (
                self._encode_pair(
                    context,
                    continuation,
                )
            )

            if not continuation_ids:
                results.append((0.0, True))
                encoded_requests.append(None)
                continue

            # Keep the whole continuation.
            max_context = (
                self.max_length
                - len(continuation_ids)
            )

            if max_context < 0:
                continuation_ids = (
                    continuation_ids[
                        : self.max_length
                    ]
                )
                max_context = 0

            if len(context_ids) > max_context:
                context_ids = context_ids[
                    -max_context:
                ]

            input_ids = (
                context_ids
                + continuation_ids
            )

            encoded_requests.append(
                (
                    input_ids,
                    len(context_ids),
                    len(continuation_ids),
                )
            )

            results.append(None)

        active = [
            (
                index,
                item,
            )
            for index, item
            in enumerate(encoded_requests)
            if item is not None
        ]

        # Group into batches.
        for start in range(
            0,
            len(active),
            self._batch_size,
        ):
            batch_items = active[
                start : start + self._batch_size
            ]

            max_len = max(
                len(item[1][0])
                for item in batch_items
            )

            batch_ids = []

            for _, (
                input_ids,
                _,
                _,
            ) in batch_items:

                padded = (
                    input_ids
                    + [0]
                    * (
                        max_len
                        - len(input_ids)
                    )
                )

                batch_ids.append(padded)

            input_tensor = torch.tensor(
                batch_ids,
                dtype=torch.long,
                device=self._device,
            )

            logits = self.model(
                input_tensor
            )

            log_probs = torch.log_softmax(
                logits,
                dim=-1,
            )

            for row, (
                original_index,
                (
                    input_ids,
                    context_len,
                    continuation_len,
                ),
            ) in enumerate(batch_items):

                # logits position t predicts
                # input token t+1.
                start_pos = (
                    context_len - 1
                )

                target_start = (
                    context_len
                )

                target_end = (
                    target_start
                    + continuation_len
                )

                if context_len == 0:
                    start_pos = 0
                    target_start = 0

                target_positions = torch.arange(
                    start_pos,
                    start_pos
                    + continuation_len,
                    device=self._device,
                )

                target_tokens = torch.tensor(
                    input_ids[
                        target_start:target_end
                    ],
                    dtype=torch.long,
                    device=self._device,
                )

                selected = log_probs[
                    row,
                    target_positions,
                    target_tokens,
                ]

                total_logprob = selected.sum().item()

                greedy_tokens = torch.argmax(
                    logits[
                        row,
                        target_positions,
                    ],
                    dim=-1,
                )

                is_greedy = bool(
                    torch.equal(
                        greedy_tokens,
                        target_tokens,
                    )
                )

                results[
                    original_index
                ] = (
                    total_logprob,
                    is_greedy,
                )

        return results

    def loglikelihood_rolling(
        self,
        requests,
        disable_tqdm=False,
    ):
        raise NotImplementedError(
            "Rolling likelihood is not needed "
            "for the selected benchmarks."
        )

    def generate_until(
        self,
        requests,
        disable_tqdm=False,
    ):
        raise NotImplementedError(
            "Generation evaluation is not needed "
            "for the selected benchmarks."
        )


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    lm = MicroLLMEval(
        checkpoint_path=CHECKPOINT_PATH,
        batch_size=8,
        device=device,
    )

    results = simple_evaluate(
        model=lm,
        tasks=[
            "hellaswag",
            "arc_easy",
            "piqa",
            "winogrande",
        ],
        num_fewshot=0,
        batch_size=8,
        device=device,
        limit=2,
    )

    print()
    print("=" * 50)
    print("MICROLLM BENCHMARK TEST")
    print("=" * 50)

    for task, values in results[
        "results"
    ].items():

        print()
        print(task)

        for metric, value in values.items():
            if isinstance(value, (int, float)):
                print(
                    f"  {metric}: {value}"
                )

    print()
    print("Task versions:")
    print(results["versions"])


if __name__ == "__main__":
    main()
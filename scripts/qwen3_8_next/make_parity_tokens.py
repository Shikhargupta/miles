"""Freeze a token sequence for the Megatron-vs-sglang logprob comparison.

Both sides must score the identical ids, so the sequence is written once to a file
rather than re-tokenized on each side. Content is chosen to exercise the parts most
likely to diverge: an EOS in the middle (the n-gram hash resets there, so PLE's
context handling is on the hook), a long enough span that the QSA indexer selects
fewer blocks than its budget for early tokens and more later, and a mix of prose
and code so the MoE router does not sit on one expert.
"""

import argparse
import json

import torch
from transformers import AutoTokenizer


TEXT_A = (
    "The compressed key for a block is the mean of its member keys, normalized and "
    "rotated at the block position. A query may attend to a block only once that "
    "block lies entirely at or before it, which is what keeps the selection causal "
    "even though the scoring runs over the whole compressed sequence. "
)
TEXT_B = (
    "def fib(n: int) -> int:\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=6, help="how many times to repeat the pair")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.hf_checkpoint, trust_remote_code=True)
    # This checkpoint has two end tokens with different meanings, and the PLE hash
    # cares about the document one: config.text_config.eos_token_id is 248044
    # (<|endoftext|>) while tokenizer.eos_token_id is 248046 (<|im_end|>, a chat-turn
    # end). sglang hashes with the config value (self.eos_token_id =
    # int(config.eos_token_id)), so a sequence built around the tokenizer's token
    # would never exercise the hash's document-boundary reset at all.
    cfg = json.load(open(f"{a.hf_checkpoint}/config.json"))
    eos = int(cfg["text_config"]["eos_token_id"])
    ids: list[int] = []
    for i in range(a.repeat):
        ids += tok(TEXT_A, add_special_tokens=False)["input_ids"]
        # A document boundary in the middle: the n-gram hash resets at EOS, so this
        # is what exercises PLE's shift_right_ignore_eos on the training side.
        if i % 2 == 1:
            ids.append(eos)
        ids += tok(TEXT_B, add_special_tokens=False)["input_ids"]

    t = torch.tensor(ids, dtype=torch.long)
    torch.save(t, a.out)
    print(f"PARITY_TOKENS n={t.numel()} eos_id={eos} eos_count={(t == eos).sum().item()}")
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()

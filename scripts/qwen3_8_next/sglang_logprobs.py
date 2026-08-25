"""Per-token logprobs from a running sglang server, for parity against Megatron.

The other half of the comparison in megatron_logprobs.py. Uses the native
/generate endpoint with return_logprob so the numbers come from the same path a
rollout takes, and max_new_tokens=0 so nothing is sampled -- we only want the
prompt's own token logprobs.

No sglang import: this talks HTTP, so the training repo takes no dependency on it.
"""

import argparse
import json
import urllib.request

import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--tokens", required=True, help=".pt file with a 1-D int64 tensor of ids")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse()
    ids = torch.load(args.tokens).to(torch.long).tolist()

    body = json.dumps(
        {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
    ).encode()
    req = urllib.request.Request(
        args.url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())

    meta = out["meta_info"] if isinstance(out, dict) else out[0]["meta_info"]
    # input_token_logprobs is [(logprob, token_id, text), ...]; the first entry has
    # logprob None because nothing precedes it.
    entries = meta["input_token_logprobs"]
    lp = [e[0] for e in entries]
    tok = [e[1] for e in entries]
    assert tok == ids, "sglang echoed different token ids than we sent"
    logprobs = torch.tensor([x for x in lp[1:]], dtype=torch.float32)

    torch.save({"input_ids": torch.tensor(ids), "logprobs": logprobs}, args.out)
    print(f"SGLANG_LOGPROBS_OK n={logprobs.numel()} mean={logprobs.mean():.4f}")


if __name__ == "__main__":
    main()

"""Compare Megatron vs sglang per-token logprobs.

This is the train/inference-consistency number the whole port is aimed at. The
target is a mean absolute difference around 1e-2; anything an order of magnitude
worse points at a real defect (a wrong weight mapping, a hyper-connection that
never got built, a layernorm still at its init value, a PLE hash off by one)
rather than at accumulated bf16 rounding.
"""

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--megatron", required=True)
    p.add_argument("--sglang", required=True)
    p.add_argument("--tol", type=float, default=2e-2)
    a = p.parse_args()

    m = torch.load(a.megatron)
    s = torch.load(a.sglang)
    assert torch.equal(m["input_ids"], s["input_ids"]), "different token ids on the two sides"

    # reshape(-1): compute_log_probs returns [T, 1] (it works in Megatron's
    # [s, b, v] layout internally), so without this the diff is 2-D and argsort
    # yields row indices.
    x, y = m["logprobs"].float().reshape(-1), s["logprobs"].float().reshape(-1)
    n = min(x.numel(), y.numel())
    x, y = x[:n], y[:n]
    d = (x - y).abs()

    print(f"tokens compared      : {n}")
    print(f"mean |diff|          : {d.mean():.6f}")
    print(f"max  |diff|          : {d.max():.6f}")
    print(f"p99  |diff|          : {d.quantile(0.99):.6f}")
    print(f"megatron mean logprob: {x.mean():.6f}")
    print(f"sglang   mean logprob: {y.mean():.6f}")
    worst = d.argsort(descending=True)[:5]
    print("worst positions:")
    for i in worst.tolist():
        print(f"  [{i:5d}] megatron={x[i]:+.5f} sglang={y[i]:+.5f} diff={d[i]:.5f}")
    print()
    print("VERDICT=" + ("PASS" if d.mean().item() < a.tol else "FAIL"))


if __name__ == "__main__":
    main()

"""Minimal repro for the attempt-21 train-step segfault.

Suspect: megatron's HybridDeviceOptimizer (--optimizer-cpu-offload
--use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d) on GB300
(aarch64 Grace + torch 2.13). In the e2e run all four last-PP-stage ranks
took a SIGSEGV (tvm_ffi's global handler caught it) ~46 min into the first
train step, right where their optimizer d2h overlap goes full-steam. The
offline bwd parity harness never ran an optimizer step, so this path was
never exercised outside the pipeline.

This builds a single-rank megatron DDP model with the same poison
ingredients as Qwen3.8-Next:
  * bf16 params with --accumulate-allreduce-grads-in-fp32,
  * one param forced to fp32 after wrap (mark_param_dtype / A_log pattern),
  * a few GB of params so the d2h overlap actually streams,
and runs fwd/bwd/step in a loop. Crash = culprit confirmed.

Run (inside the training container, 1 GPU):
  torchrun --nproc-per-node 1 repro_hdo_segfault.py [--no-offload]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-offload", action="store_true", help="control run: plain GPU adam")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--no-fp32-island", action="store_true", help="skip the marked-fp32 param")
    cli = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29399")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(0)

    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer import TransformerConfig

    parallel_state.initialize_model_parallel()

    hidden, layers = cli.hidden, cli.layers

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(layers))
            # A_log-style fp32 island in a bf16 model
            self.a_log = nn.Parameter(torch.randn(64))

        def forward(self, x):
            for l in self.layers:
                x = torch.relu(l(x))
            return (x.float().pow(2).mean() + self.a_log.float().sum() * 0.0).to(torch.float32)

    model = Toy().cuda().to(torch.bfloat16)
    if not cli.no_fp32_island:
        model.a_log.data = model.a_log.data.to(torch.float32)  # post-cast, param identity kept

    # minimal config objects (mirrors the e2e's relevant knobs)
    tf_cfg = TransformerConfig(
        num_layers=1, hidden_size=hidden, num_attention_heads=8,
        bf16=True, params_dtype=torch.bfloat16,
    )
    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,          # --accumulate-allreduce-grads-in-fp32
        overlap_grad_reduce=False,
        use_distributed_optimizer=True,
    )
    model = DistributedDataParallel(tf_cfg, ddp_cfg, model)

    opt_cfg = OptimizerConfig(
        optimizer="adam", lr=1e-6, weight_decay=0.1,
        adam_beta1=0.9, adam_beta2=0.98,
        bf16=True, params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        optimizer_cpu_offload=not cli.no_offload,
        use_precision_aware_optimizer=not cli.no_offload,
        overlap_cpu_optimizer_d2h_h2d=not cli.no_offload,
    )
    opt = get_megatron_optimizer(opt_cfg, [model])
    nparams = sum(p.numel() for p in model.parameters())
    print(f"REPRO start: offload={not cli.no_offload} params={nparams/1e9:.2f}B "
          f"({nparams*2/2**30:.1f} GiB bf16)", flush=True)

    x = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16)
    for it in range(cli.steps):
        model.zero_grad_buffer()
        opt.zero_grad()
        loss = model(x)
        loss.backward()
        ok, gnorm, _ = opt.step()
        torch.cuda.synchronize()
        print(f"step {it}: loss={loss.item():.5f} ok={ok} grad_norm={gnorm}", flush=True)
    print("REPRO_SURVIVED", flush=True)


if __name__ == "__main__":
    main()

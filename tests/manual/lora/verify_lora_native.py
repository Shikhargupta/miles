"""Numerical verification of native (raw-mode) LoRA against dense reference math.

Builds a small real megatron-core GPTModel, attaches adapters with
``apply_native_lora``, and checks the adapter branch against independently
computed dense math at TP1 / TP2 / TP2+sequence-parallel.

Run it directly (needs as many GPUs as --tp):

  PYTHONPATH=/root/Megatron-LM:. torchrun --nproc-per-node 1 \
      tests/manual/lora/verify_lora_native.py --tp 1
  PYTHONPATH=/root/Megatron-LM:. torchrun --nproc-per-node 2 \
      tests/manual/lora/verify_lora_native.py --tp 2 --sp

Exits nonzero if any check fails. Checks, per configuration:

  1. no-op: a fresh adapter (B is zero-init) leaves the output bit-identical
  2. delta: the adapter branch equals scale * B @ (A @ x) computed densely from the
     TP-gathered adapter, for both a column-parallel (fc1) and a row-parallel (fc2)
     module -- the base GEMM is subtracted out, so this tests only our math
  3. export: TP shards gather to tensors identical on every rank
  4. round-trip: export -> load into a fresh model reproduces params and outputs
  5. grads: dL/dA == 0 while dL/dB != 0 for a fresh adapter (B zero-init), grads are
     nonzero once B is randomized, and replicated-param grads agree across TP after
     reduce_marked_lora_grads combined genuinely distinct per-rank partials
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.megatron_utils.lora_native import (
    NativeLoRAAdapter,
    _rmsnorm,
    apply_native_lora,
    export_lora_hf_named,
    load_lora_adapter_hf,
)
from miles.backends.megatron_utils.lora_utils import reduce_marked_lora_grads

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    if dist.get_rank() == 0:
        print(f"[{tag}] {name} {detail}", flush=True)


def build(tp, seq_parallel, seed=1234):
    torch.manual_seed(seed)
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=4,  # GQA, >= tp
        ffn_hidden_size=512,
        kv_channels=32,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp,
        sequence_parallel=seq_parallel,
        bf16=False,
        params_dtype=torch.float32,
        gated_linear_unit=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        pipeline_dtype=torch.float32,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=None, moe_grouped_gemm=False)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=512,
        max_sequence_length=64,
        pre_process=True,
        post_process=True,
    ).cuda()
    return model, cfg


class Args:
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.0
    lora_A_init_method = "xavier"
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_provider_path = None


def fwd(model, tokens, pos, mask):
    return model(input_ids=tokens, position_ids=pos, attention_mask=mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--sp", action="store_true")
    a = p.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.initialize_model_parallel(tensor_model_parallel_size=a.tp)
    model_parallel_cuda_manual_seed(1234)
    label = f"TP{a.tp}{'+SP' if a.sp else ''}"

    torch.manual_seed(7)
    b, s = 2, 16
    tokens = torch.randint(0, 512, (b, s), device="cuda")
    pos = torch.arange(s, device="cuda").unsqueeze(0).expand(b, s)
    mask = torch.ones(b, 1, s, s, dtype=torch.bool, device="cuda").tril().logical_not()

    # ---- 1. fresh adapter is a no-op -------------------------------------
    lora_model, _ = build(a.tp, a.sp)
    lora_model.eval()
    with torch.no_grad():
        out_base = fwd(lora_model, tokens, pos, mask).clone()

    # Capture per-module base outputs BEFORE wrapping, so the delta check below
    # compares only the adapter branch and never has to reproduce TE's base GEMM.
    layer0 = lora_model.decoder.layers[0]
    fc1_mod, fc2_mod = layer0.mlp.linear_fc1, layer0.mlp.linear_fc2
    torch.manual_seed(31337)
    x_fc1 = torch.randn(4, 1, fc1_mod.weight.shape[1], device="cuda")
    x_fc2 = torch.randn(4, 1, fc2_mod.weight.shape[1], device="cuda")
    with torch.no_grad():
        y0_fc1 = fc1_mod(x_fc1)[0].clone()
        y0_fc2 = fc2_mod(x_fc2)[0].clone()

    apply_native_lora(lora_model, Args())
    with torch.no_grad():
        out_fresh = fwd(lora_model, tokens, pos, mask)
    check(
        f"{label} fresh adapter is exact no-op",
        torch.equal(out_base, out_fresh),
        f"max|d|={(out_base - out_fresh).abs().max().item():.3e}",
    )

    n_adapters = sum(1 for m in lora_model.modules() if isinstance(m, NativeLoRAAdapter))
    check(f"{label} adapters attached", n_adapters == 2 * 4, f"count={n_adapters} (expect 8)")

    trainable = [n for n, q in lora_model.named_parameters() if q.requires_grad]
    check(
        f"{label} only adapter params trainable",
        all("lora" in n for n in trainable) and len(trainable) > 0,
        f"n_trainable={len(trainable)}",
    )

    # ---- 2. adapter delta matches an independent dense reference ----------
    torch.manual_seed(99)
    for m in lora_model.modules():
        if isinstance(m, NativeLoRAAdapter):
            for _, prm in m.named_parameters(recurse=False):
                with torch.no_grad():
                    prm.normal_(0, 0.02)

    scale = Args.lora_alpha / Args.lora_rank
    tp_group = ps.get_tensor_model_parallel_group()

    def gather_cat(t, dim):
        parts = [torch.empty_like(t) for _ in range(a.tp)]
        dist.all_gather(parts, t.contiguous(), group=tp_group)
        return torch.cat(parts, dim=dim)

    # column-parallel fc1: A replicated, B row-sharded -> local delta uses local B.
    ad1 = layer0.mlp.lora_fc1_adapter
    with torch.no_grad():
        got1 = fc1_mod(x_fc1)[0] - y0_fc1
        xn = _rmsnorm(x_fc1, fc1_mod.layer_norm_weight, lora_model.config.layernorm_epsilon)
        if a.sp:
            xn = gather_cat(xn, 0)
        ref1 = scale * torch.cat(
            [F.linear(F.linear(xn, ad1.gate_A), ad1.gate_B), F.linear(F.linear(xn, ad1.up_A), ad1.up_B)], dim=-1
        )
    e1 = (got1 - ref1).abs().max().item()
    r1 = e1 / max(ref1.abs().max().item(), 1e-9)
    check(f"{label} column-parallel (fc1) delta == dense reference", r1 < 1e-5, f"max|d|={e1:.3e} rel={r1:.2e}")

    # row-parallel fc2: A column-sharded + TP-reduced -> reference from FULL A and x.
    ad2 = layer0.mlp.lora_fc2_adapter
    with torch.no_grad():
        got2 = fc2_mod(x_fc2)[0] - y0_fc2
        # sum_j A_j @ x_j == A_full @ x_full; under SP that sum is reduce-SCATTERED,
        # so this rank keeps only its sequence shard before applying B.
        s_full = F.linear(gather_cat(x_fc2, -1), gather_cat(ad2.down_A, 1))
        if a.sp:
            s_full = s_full.chunk(a.tp, dim=0)[ps.get_tensor_model_parallel_rank()]
        ref2 = scale * F.linear(s_full, ad2.down_B)
    e2 = (got2 - ref2).abs().max().item()
    r2 = e2 / max(ref2.abs().max().item(), 1e-9)
    check(f"{label} row-parallel (fc2) delta == dense reference", r2 < 1e-5, f"max|d|={e2:.3e} rel={r2:.2e}")

    # ---- 3/4. export: agreement across TP + round-trip through load ------
    exported = export_lora_hf_named([lora_model])
    check(f"{label} export covers all adapters", len(exported) == 2 * 14, f"n={len(exported)} (expect 28)")

    flat = torch.cat([t.float().reshape(-1) for _, t in exported])
    gathered = [torch.empty_like(flat) for _ in range(a.tp)]
    dist.all_gather(gathered, flat, group=ps.get_tensor_model_parallel_group())
    same = all(torch.equal(gathered[0], g) for g in gathered)
    check(f"{label} exported adapter identical on every TP rank", same)

    import json
    import tempfile

    from safetensors.torch import save_file

    tmp = tempfile.mkdtemp()
    if dist.get_rank() == 0:
        save_file({n: t.contiguous() for n, t in exported}, os.path.join(tmp, "adapter_model.safetensors"))
        json.dump({"r": Args.lora_rank}, open(os.path.join(tmp, "adapter_config.json"), "w"))
    obj = [tmp]
    dist.broadcast_object_list(obj, src=0)
    tmp = obj[0]
    dist.barrier()

    fresh, _ = build(a.tp, a.sp)
    apply_native_lora(fresh, Args())
    # copy the base (non-adapter) weights so only the adapter differs
    base_state = {k: v for k, v in lora_model.state_dict().items() if "lora" not in k}
    missing, unexpected = fresh.load_state_dict(base_state, strict=False)
    check(
        f"{label} base weights copied for round-trip",
        not unexpected and all("lora" in m for m in missing),
        f"missing={len(missing)} unexpected={len(unexpected)}",
    )
    n_loaded = load_lora_adapter_hf([fresh], tmp)
    check(f"{label} load consumed every adapter tensor", n_loaded == 2 * 14, f"loaded={n_loaded}")

    max_d = 0.0
    for (n1, p1), (n2, p2) in zip(
        sorted((n, q) for n, q in lora_model.named_parameters() if "lora" in n),
        sorted((n, q) for n, q in fresh.named_parameters() if "lora" in n),
        strict=True,
    ):
        assert n1 == n2, (n1, n2)
        max_d = max(max_d, (p1.float() - p2.float()).abs().max().item())
    # export casts to bf16, so the round-trip tolerance is bf16 resolution
    check(f"{label} export->load round-trip preserves params", max_d < 2e-2, f"max|d|={max_d:.3e}")

    fresh.eval()
    lora_model.eval()
    with torch.no_grad():
        o1 = fwd(lora_model, tokens, pos, mask)
        o2 = fwd(fresh, tokens, pos, mask)
    d = (o1 - o2).abs().max().item()
    check(f"{label} round-tripped model reproduces outputs", d < 5e-2, f"max|d|={d:.3e}")

    # ---- 5. gradients: nonzero, TP-consistent, and honor B's zero init ----
    # Fresh-adapter invariant: with B == 0, dL/dA == 0 and dL/dB != 0. Check that
    # first on a clean model, then check the general case on the randomized one.
    fresh2, _ = build(a.tp, a.sp)
    apply_native_lora(fresh2, Args())
    fresh2.train()
    fwd(fresh2, tokens, pos, mask).square().mean().backward()
    a_grads, b_grads = [], []
    for n, q in fresh2.named_parameters():
        if not q.requires_grad or q.grad is None:
            continue
        (a_grads if n.endswith("_A") else b_grads).append(q.grad.abs().max().item())
    check(
        f"{label} fresh adapter: dL/dA == 0 (B is zero-init)",
        a_grads and max(a_grads) == 0.0,
        f"max|dA|={max(a_grads):.3e}",
    )
    check(
        f"{label} fresh adapter: dL/dB != 0",
        b_grads and min(b_grads) > 0.0,
        f"min|dB|={min(b_grads):.3e} max|dB|={max(b_grads):.3e}",
    )

    lora_model.train()
    out = fwd(lora_model, tokens, pos, mask)
    out.square().mean().backward()
    for prm in lora_model.parameters():
        if prm.requires_grad and prm.grad is not None:
            prm.main_grad = prm.grad
    tagged = [
        (n, q)
        for n, q in lora_model.named_parameters()
        if getattr(q, "_lora_grad_sum_group", None) == "tp" and q.grad is not None
    ]
    check(f"{label} tagged replicated params exist", len(tagged) > 0, f"n={len(tagged)}")
    pre = {n: q.main_grad.clone() for n, q in tagged}
    reduce_marked_lora_grads([lora_model])

    nonzero = all(q.main_grad.abs().max().item() > 0 for _, q in tagged)
    check(f"{label} adapter grads are nonzero (randomized B)", nonzero)

    ok, worst = True, 0.0
    for _, q in tagged:
        parts = [torch.empty_like(q.main_grad) for _ in range(a.tp)]
        dist.all_gather(parts, q.main_grad, group=tp_group)
        for pg in parts:
            worst = max(worst, (pg - parts[0]).abs().max().item())
        ok = ok and all(torch.allclose(pg, parts[0], atol=1e-5) for pg in parts)
    check(f"{label} replicated-param grads consistent across TP", ok, f"max spread={worst:.3e}")

    if a.tp > 1:
        # The sum must actually change the grads: partials differ per rank.
        changed = any(not torch.allclose(pre[n], q.main_grad) for n, q in tagged)
        check(f"{label} TP sum actually combined distinct partial grads", changed)

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"\n=== {label}: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} ===", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if FAILS else 0)


main()

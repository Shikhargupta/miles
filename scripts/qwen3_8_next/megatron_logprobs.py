"""Per-token logprobs from the Megatron model, for parity against sglang.

This is the measurement the whole exercise is aimed at: a mapping error, a missing
hyper-connection, a layernorm left at its init value, or an off-by-one in the PLE
hash all show up here as logprobs that do not match the rollout engine. A
tensor-by-tensor round trip through the bridge cannot catch any of them -- it
writes and reads with the same mapping, so it is close to a tautology.

Run under torchrun with the same parallelism the checkpoint was saved with.
Writes {out}/megatron_logprobs.pt with the token ids and per-token logprobs so the
comparison itself stays out of this process.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _converter_arg_provider():
    """Reuse tools/convert_hf_to_torch_dist.py's argument group.

    miles' get_model_provider_func and init() read fields that only the converter's
    parser adds (--hf-checkpoint, --megatron-to-hf-mode,
    --custom-model-provider-path). Reimplementing them here means rediscovering each
    one through an AttributeError on a separate run, and drifting the moment the
    converter gains another. Loading the real function by path keeps them in step.

    tools/ is not a package, so this goes through importlib rather than an import.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tools" / "convert_hf_to_torch_dist.py"
    spec = importlib.util.spec_from_file_location("_miles_convert_args", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add_conversion_args


def add_parity_args(parser):
    """Register the converter's flags plus ours with Megatron's parser.

    A separate ArgumentParser does not work: Megatron's parse_args() runs on the
    same argv and rejects anything it does not recognise, so everything has to go
    through extra_args_provider.
    """
    parser = _converter_arg_provider()(parser)
    group = parser.add_argument_group(title="qwen3.8-next parity")
    group.add_argument("--tokens", type=str, required=True,
                       help=".pt file with a 1-D int64 tensor of input ids")
    group.add_argument("--parity-out", type=str, required=True)
    group.add_argument(
        "--dump",
        action="store_true",
        help="dump the hyper-connection residual stream at every layer boundary via "
             "the sglang dumper, for comparison against the reference implementation",
    )
    group.add_argument(
        "--backward",
        action="store_true",
        help="run forward+backward with a fixed CE loss and dump gradients via the "
             "sglang dumper (DUMPER_ENABLE_GRAD=1); for run-to-run gradient parity",
    )
    group.add_argument(
        "--trace",
        action="store_true",
        help="dump per-layer activation norms, logits stats and suspicious parameter "
             "stats; for localising a structurally wrong forward",
    )
    return parser


def _find_ple_embedding(module):
    """The PLE table hangs off the attention hyper-connection of one layer."""
    from miles_plugins.models.qwen3_8_next.ops.ple_embedding import (
        Qwen38NextFrozenNGramEmbedding,
    )

    for m in module.modules():
        if isinstance(m, Qwen38NextFrozenNGramEmbedding):
            return m
    return None


# (megatron suffix, hf name, dim sharded across TP or None). Deliberately hand
# written and short: these are the tensors whose being wrong makes the output
# uniform noise no matter how healthy every layer looks, and the point is to
# compare against the released checkpoint rather than against another copy of our
# own conversion.
_KEY_TENSORS = [
    ("embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight", 0),
    ("output_layer.weight", "lm_head.weight", 0),
    ("decoder.final_layernorm.weight", "model.language_model.norm.weight", None),
    ("decoder.layers.0.self_attention_hyper_connection.hc_norm.weight",
     "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight", None),
    # Layer 0's gated-delta-net. miles' Qwen3.5 wrapper runs the linear attention
    # duplicated on every TP rank rather than sharded, so all of these must be the
    # full HF tensor on every rank; a shard here is the whole bug.
    ("decoder.layers.0.self_attention.linear_attn.in_proj_qkv.weight",
     "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", None),
    ("decoder.layers.0.self_attention.linear_attn.in_proj_z.weight",
     "model.language_model.layers.0.linear_attn.in_proj_z.weight", None),
    ("decoder.layers.0.self_attention.linear_attn.in_proj_a.weight",
     "model.language_model.layers.0.linear_attn.in_proj_a.weight", None),
    ("decoder.layers.0.self_attention.linear_attn.out_proj.weight",
     "model.language_model.layers.0.linear_attn.out_proj.weight", None),
    ("decoder.layers.0.self_attention.linear_attn.conv1d.weight",
     "model.language_model.layers.0.linear_attn.conv1d.weight", None),
    # A_log and dt_bias are the ones that matter most here: the module initialises
    # A_log from torch.empty().uniform_(0, 16) and dt_bias from torch.ones, so an
    # unmapped tensor stays a plausible-looking random value rather than an obvious
    # zero, and the decay gate g = -exp(A_log) * softplus(a + dt_bias) is silently
    # garbage while every shape and every other weight checks out.
    ("decoder.layers.0.self_attention.linear_attn.A_log",
     "model.language_model.layers.0.linear_attn.A_log", None),
    ("decoder.layers.0.self_attention.linear_attn.dt_bias",
     "model.language_model.layers.0.linear_attn.dt_bias", None),
    ("decoder.layers.0.self_attention.linear_attn.in_proj_b.weight",
     "model.language_model.layers.0.linear_attn.in_proj_b.weight", None),
    ("decoder.layers.0.self_attention.linear_attn.norm.weight",
     "model.language_model.layers.0.linear_attn.norm.weight", None),
    # The PLE at layer 1 (hf index 1). Its projections are duplicated across TP,
    # so all of these are the full HF tensor on every rank.
    ("layers.1.self_attention_hyper_connection.ple.key_proj.weight",
     "model.language_model.layers.1.ple.key_proj.weight", None),
    ("layers.1.self_attention_hyper_connection.ple.value_proj.weight",
     "model.language_model.layers.1.ple.value_proj.weight", None),
    ("layers.1.self_attention_hyper_connection.ple.norm_key",
     "model.language_model.layers.1.ple.norm_key.weight", None),
    ("layers.1.self_attention_hyper_connection.ple.norm_query",
     "model.language_model.layers.1.ple.norm_query.weight", None),
    ("layers.1.self_attention_hyper_connection.ple.norm_conv",
     "model.language_model.layers.1.ple.norm_conv.weight", None),
    ("layers.1.self_attention_hyper_connection.ple.conv1d_weight",
     "model.language_model.layers.1.ple.conv1d.weight", None),
]


def _check_key_weights(module, hf_checkpoint: str) -> None:
    """Compare a few decisive tensors against the released checkpoint."""
    import json as _json

    from megatron.core import parallel_state as mpu
    from safetensors import safe_open

    if dist.get_rank() != 0:
        return
    index = _json.load(open(f"{hf_checkpoint}/model.safetensors.index.json"))["weight_map"]
    params = dict(module.named_parameters())
    tp = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()

    print("=== key weights vs the HF checkpoint ===", flush=True)
    for suffix, hf_name, shard_dim in _KEY_TENSORS:
        got = next((v for k, v in params.items() if k.endswith(suffix)), None)
        if got is None:
            print(f"  MISSING in model: {suffix}", flush=True)
            continue
        if hf_name not in index:
            print(f"  MISSING in hf: {hf_name}", flush=True)
            continue
        with safe_open(f"{hf_checkpoint}/{index[hf_name]}", framework="pt") as f:
            ref = f.get_tensor(hf_name)
        if shard_dim is not None and tp > 1:
            n = ref.shape[shard_dim] // tp
            ref = ref.narrow(shard_dim, tp_rank * n, n)
        mine = got.detach().float().cpu()
        ref = ref.float()
        if mine.shape != ref.shape:
            print(f"  SHAPE  {suffix}: mine={tuple(mine.shape)} hf={tuple(ref.shape)}", flush=True)
            continue
        d = (mine - ref).abs()
        verdict = "EQUAL" if d.max().item() < 1e-3 else "DIFFERENT"
        print(f"  {verdict:<9} {suffix:<62} max|d|={d.max():.5f} "
              f"mine_std={mine.std():.5f} hf_std={ref.std():.5f}", flush=True)


def _check_param_replication(module) -> None:
    """List parameters that are not identical on every tensor-parallel rank.

    The GDN linear attention and the hyper-connection gates run duplicated on all
    TP ranks, so their weights must match everywhere; a shard there makes each rank
    compute a different residual stream while every individual tensor still looks
    correct on rank 0. Genuinely TP-sharded tensors (attention qkv, expert MLPs)
    are expected in this list, so it is read by looking for what should NOT be.
    """
    from megatron.core import parallel_state as mpu

    group = mpu.get_tensor_model_parallel_group()
    if mpu.get_tensor_model_parallel_world_size() == 1:
        return

    differing = []
    for name, prm in module.named_parameters():
        with torch.no_grad():
            mine = prm.detach().float()
            ref = mine.clone()
            dist.broadcast(ref, src=dist.get_global_rank(group, 0), group=group)
            d = (mine - ref).abs().max()
            # Reduce over the group before reading it: rank 0 is the reference, so
            # measuring only there reports zero for every tensor no matter how the
            # other ranks differ.
            dist.all_reduce(d, op=dist.ReduceOp.MAX, group=group)
            d = d.item()
        if d > 0:
            differing.append((name, d))

    if dist.get_rank() != 0:
        return
    print(f"=== parameters differing across TP ranks: {len(differing)} ===", flush=True)
    interesting = [
        (n, d) for n, d in differing
        if "linear_attn" in n or "hyper_connection" in n or "ple" in n or "indexer" in n
    ]
    for n, d in interesting:
        print(f"  SHOULD-BE-REPLICATED {n:<74} max|d|={d:.5f}", flush=True)
    if not interesting:
        print("  (none of the replicated-by-design modules differ)", flush=True)
    print(f"  plus {len(differing) - len(interesting)} genuinely TP-sharded tensors", flush=True)


class _RepeatBisect:
    """Find the first module whose output differs between two identical forwards.

    Pass 1 records every module's output; pass 2 compares in forward order and
    reports the earliest mismatches. Parameters and buffers are already proven
    unchanged, so whatever differs first is reading state torch does not track
    (allocator garbage, a custom kernel's scratch, a cache keyed on something
    unstable) -- and naming the module names the suspect.
    """

    def __init__(self, module):
        self.pass_no = 1
        self.recorded = {}
        self.mismatches = []
        self.order = []
        for name, mod in module.named_modules():
            if name:
                mod.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t) or not t.is_floating_point():
                return
            if self.pass_no == 1:
                if name not in self.recorded:  # keep the first call only
                    self.recorded[name] = t.detach().clone()
                    self.order.append(name)
            else:
                ref = self.recorded.get(name)
                if ref is None or ref.shape != t.shape:
                    return
                if name in {m for m, _ in self.mismatches}:
                    return
                d = (t.detach().float() - ref.float()).abs().max().item()
                if d > 0:
                    self.mismatches.append((name, d))
        return fn

    def report(self):
        if dist.get_rank() != 0:
            return
        rank_of = {n: i for i, n in enumerate(self.order)}
        first = sorted(self.mismatches, key=lambda x: rank_of.get(x[0], 1 << 30))
        print(f"REPEAT_BISECT mismatching_modules={len(first)} of {len(self.order)}", flush=True)
        for n, d in first[:15]:
            print(f"  DIFFERS[{rank_of.get(n, -1):4d}] {n:<86} max|d|={d:.6g}", flush=True)
        clean_until = first[0][0] if first else None
        if clean_until is not None:
            i = rank_of[clean_until]
            print(f"  last clean before it: "
                  f"{self.order[i - 1] if i > 0 else '<none>'}", flush=True)


def _pin_chunk_o_config() -> None:
    """Pin chunk_fwd_o to a single autotune config (PARITY_CHUNK_O_BK).

    The kernel has three candidate configs and triton picks by measured timing,
    so the choice depends on GPU load at first call -- which is why the isolated
    benchmark was bit-stable and the in-model run is not. Running the model once
    per pinned config identifies whether one of them is miscompiled on this
    hardware rather than the kernel being racy per se.
    """
    bk = int(os.environ["PARITY_CHUNK_O_BK"])
    import fla.ops.common.chunk_o as co

    # The decorator stack is Heuristics(Autotuner(JITFunction)); only the
    # Autotuner layer has .configs, so walk .fn down to it.
    tuner = co.chunk_fwd_kernel_o
    while not hasattr(tuner, "configs"):
        tuner = tuner.fn
    kept = [c for c in tuner.configs if c.kwargs.get("BK") == bk]
    assert kept, f"no chunk_fwd_o config with BK={bk}"
    tuner.configs = kept
    print(f"CHUNK_O_PINNED BK={bk} configs={len(kept)}", flush=True)


def _wrap_fla_subkernels() -> None:
    """Double-call every sub-kernel of fla's chunk forward and compare in place.

    The chunk forward is l2norm -> cumsum -> kkt -> solve_tril -> w,u -> fwd_h ->
    fwd_o; the module-level probe showed the composite diverging on identical
    inputs, so this names which stage. Patching the chunk module's imported
    symbols covers the actual call sites. Debug-only (PARITY_FLA_SUBPROBE=1).
    """
    import fla.ops.gated_delta_rule.chunk as fc

    def wrap(fn, name):
        state = {"reported": 0}

        def double(*args, **kwargs):
            o1 = fn(*args, **kwargs)
            o2 = fn(*args, **kwargs)

            def flat(o):
                if isinstance(o, tuple):
                    return [t for t in o if torch.is_tensor(t)]
                return [o] if torch.is_tensor(o) else []

            for i, (a, b) in enumerate(zip(flat(o1), flat(o2))):
                if a.shape != b.shape:
                    continue
                d = (a.float() - b.float()).abs().max().item()
                if d > 0 and state["reported"] < 3:
                    state["reported"] += 1
                    print(f"FLA_SUBPROBE {name}[out{i}] max|d|={d:.6g}", flush=True)
            return o1

        return double

    for name in (
        "l2norm_fwd",
        "chunk_local_cumsum",
        "chunk_scaled_dot_kkt_fwd",
        "solve_tril",
        "recompute_w_u_fwd",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_fwd_o",
    ):
        fn = getattr(fc, name, None)
        if fn is not None:
            setattr(fc, name, wrap(fn, name))
    print("FLA_SUBPROBE installed", flush=True)


def _wrap_chunk_double_call(module) -> None:
    """Run every GDN chunk kernel twice per call and compare in place.

    Within one call the inputs are the same objects, so any difference is the
    kernel itself misbehaving under real model load (a race or scratch reuse);
    bitwise-equal results push the suspicion back onto the inputs. Debug-only,
    activated by PARITY_CHUNK_DOUBLE=1.
    """
    wrapped = 0
    for name, mod in module.named_modules():
        fn = getattr(mod, "chunk_gated_delta_rule", None)
        if fn is None or not callable(fn):
            continue

        def make(fn, name):
            state = {"reported": False}

            def double(*args, **kwargs):
                o1 = fn(*args, **kwargs)
                o2 = fn(*args, **kwargs)
                t1 = o1[0] if isinstance(o1, tuple) else o1
                t2 = o2[0] if isinstance(o2, tuple) else o2
                d = (t1.float() - t2.float()).abs().max().item()
                if d > 0 and not state["reported"]:
                    state["reported"] = True
                    print(f"CHUNK_DOUBLE_DIFFERS {name} max|d|={d:.6g}", flush=True)
                return o1

            return double

        mod.chunk_gated_delta_rule = make(fn, name)
        wrapped += 1
    if dist.get_rank() == 0:
        print(f"CHUNK_DOUBLE wrapped={wrapped}", flush=True)


def _run_backward_parity(model, margs, input_ids, position_ids, attention_mask,
                         packed_seq_params, ngram_ids, cu_seqlens, ids):
    """One fwd+bwd with a fixed loss; gradients reach the dumper via its hooks.

    The loss is the mean token cross-entropy built from miles' own
    compute_log_probs, so the parity target is the same math the RL trainer
    differentiates. Params stay bf16 with no DDP wrap, so grads land in
    ``param.grad`` directly -- the manual's main_grad caveat applies only to
    DDP-wrapped runs.
    """
    from megatron.core import parallel_state
    from miles.backends.training_utils.loss_hub.math_utils import compute_log_probs
    from miles_plugins.models.qwen3_8_next.ops.ple_context import ple_forward_context

    try:
        from sglang.srt.debug_utils.dumper import dumper
    except Exception:
        dumper = None

    m = model[0]
    # Stays in eval(): gradients flow regardless (requires_grad + no no_grad), the
    # forward is then the exact fwd-parity path (dropout is 0 anyway), and
    # Megatron's MoE layer refuses train-mode TP without sequence parallelism.
    for prm in m.parameters():
        prm.requires_grad_(True)

    with ple_forward_context(ngram_ids, cu_seqlens=cu_seqlens):
        out = m(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
    logits = out if isinstance(out, torch.Tensor) else out[0]
    flat_logits = logits.squeeze(1) if logits.shape[1] == 1 else logits.squeeze(0)
    logprobs = compute_log_probs(
        flat_logits[:-1].to(torch.float32, copy=True),
        ids[1:],
        parallel_state.get_tensor_model_parallel_group(),
    )
    # loss = mean CE over the sequence; any other side must use this exact formula.
    loss = -logprobs.mean()
    if dumper is not None:
        dumper.dump("loss", loss)
    loss.backward()

    gsq = torch.zeros((), dtype=torch.float64, device=loss.device)
    n_with_grad = 0
    fingerprints = []
    for name, prm in m.named_parameters():
        if prm.grad is not None:
            n_with_grad += 1
            g = prm.grad.detach()
            gsq += g.double().pow(2).sum()
            # Norm + first/last element as the parity fingerprint. Dumping the full
            # grads is not viable at this size: one MoE layer's fused expert grad is
            # ~3.4 GB, x48 layers x4 ranks would be terabytes per run. Two
            # deterministic runs must match these scalars bitwise; if they do not,
            # the full-tensor comparison can be done selectively on the named param.
            flat = g.reshape(-1)
            fingerprints.append(
                (name, g.double().norm().item(), flat[0].item(), flat[-1].item())
            )
    if dumper is not None:
        dumper.dump("pgrad_fingerprints",
                    torch.tensor([[n_, f_, l_] for _, n_, f_, l_ in fingerprints],
                                 dtype=torch.float64))
    if dist.get_rank() == 0:
        for name, n_, f_, l_ in fingerprints[:6]:
            print(f"  PGRAD {name:<70} norm={n_:.8e} first={f_:.6e} last={l_:.6e}", flush=True)
    # TP-sharded grads: sum of squares over the group counts each shard once;
    # replicated modules (GDN, HC, PLE) are counted TP times, which is consistent
    # between the two runs being compared, so fine for a parity number (NOT a
    # training grad-norm).
    dist.all_reduce(gsq)
    if dumper is not None:
        dumper.step()
    if dist.get_rank() == 0:
        print(
            f"BWD_OK loss={loss.item():.6f} grad_norm={gsq.sqrt().item():.6f} "
            f"params_with_grad={n_with_grad}",
            flush=True,
        )
    dist.barrier()


def _install_dump(module) -> None:
    """Dump the per-layer residual stream under names the sglang side also uses.

    Deliberately explicit rather than the dumper's non-intrusive mode: that names
    tensors after module paths, and the two frameworks' module paths do not match
    (``decoder.layers.0.self_attention`` vs ``model.layers.0.self_attn``), while the
    comparator groups files into bundles by name. Matching names is what makes the
    comparison possible at all.

    Megatron carries [s, b, h] where sglang carries a flat [t, h]; the comparator's
    token aligner flattens the BS layout, which is why the dims differ between the
    two sides while describing the same tensor.
    """
    from sglang.srt.debug_utils.dumper import dumper

    dec = module.decoder if hasattr(module, "decoder") else module.module.decoder

    def emb_hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        dumper.dump("emb", t, dims="t 1 h # tp:replicated")

    def layer_hook(i):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            dumper.dump(f"hc_L{i:02d}", t, dims="t 1 h # tp:replicated")
        return fn

    def final_hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        dumper.dump("final_mix", t, dims="t 1 h # tp:replicated")

    emb = module.embedding if hasattr(module, "embedding") else module.module.embedding
    emb.register_forward_hook(emb_hook)
    for i, layer in enumerate(dec.layers):
        layer.register_forward_hook(layer_hook(i))

    # Sub-layer dumps for the first few layers. The residual stream must be
    # bit-identical on every TP rank; it is not, and these narrow the first place
    # where the ranks part company. Kept to a few layers because that is where the
    # divergence starts -- once ranks differ, everything after them differs too.
    def sub_hook(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                dumper.dump(name, t, dims="t 1 h # tp:replicated")
        return fn

    for i in (0, 1, 2, 3, 7):
        if i >= len(dec.layers):
            break
        layer = dec.layers[i]
        # Names are the semantic role, not the Megatron module path, so they match
        # what the sglang side dumps at the same four points in its layer.
        for attr, label in (
            ("self_attention_hyper_connection", "attn_mix"),
            ("self_attention", "attn_out"),
            ("mlp_hyper_connection", "mlp_mix"),
            ("mlp", "mlp_out"),
        ):
            mod = getattr(layer, attr, None)
            if mod is not None and not isinstance(mod, torch.nn.Identity):
                mod.register_forward_hook(sub_hook(f"L{i:02d}.{label}"))
    # final_layernorm's *input* is the contracted [s, b, 2560] stream, matching what
    # sglang dumps right after hyper_connection_mixer.mix; hook the contraction
    # instead if the block exposes one.
    if hasattr(dec, "final_layernorm") and dec.final_layernorm is not None:
        dec.final_layernorm.register_forward_pre_hook(
            lambda _m, inp: dumper.dump("final_mix", inp[0], dims="t 1 h # tp:replicated")
        )
    else:
        dec.register_forward_hook(final_hook)


def _install_trace(module) -> None:
    """Report which parameters never got real values, and where activations go bad.

    Two failure modes look identical in the logprobs -- a near-uniform output -- and
    are distinguished here. A parameter left at its init still has a tidy
    N(0, init_method_std) or all-zero/all-one signature, so scanning stats finds
    weights the checkpoint never wrote. And an activation norm that explodes or
    collapses at one layer points at that layer's wiring rather than at the weights.
    """
    import math as _math

    rank = dist.get_rank()

    if rank == 0:
        print("=== parameter stats (flagging never-loaded) ===", flush=True)
        std_init = getattr(module.config, "init_method_std", 0.02)
        flagged = 0
        for name, prm in module.named_parameters():
            with torch.no_grad():
                p32 = prm.detach().float()
                std = p32.std().item() if p32.numel() > 1 else 0.0
                mean = p32.mean().item()
                amax = p32.abs().max().item()
            why = None
            if amax == 0.0:
                why = "all zeros"
            elif p32.numel() > 1 and abs(std - std_init) / std_init < 0.05 and abs(mean) < 1e-3:
                why = f"looks like N(0,{std_init}) init"
            elif abs(mean - 1.0) < 1e-6 and std == 0.0:
                why = "all ones (untouched norm)"
            if why is not None:
                flagged += 1
                print(f"  SUSPECT {name:<78} std={std:.5f} mean={mean:+.5f} :: {why}", flush=True)
        print(f"  flagged {flagged} parameter tensors", flush=True)

    def hook(name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t):
                return
            with torch.no_grad():
                f = t.detach().float()
                rms = f.pow(2).mean().sqrt().item()
                amax = f.abs().max().item()
                bad = "  <-- non-finite" if not torch.isfinite(f).all() else ""
            if rank == 0:
                print(f"  {name:<52} shape={tuple(t.shape)} rms={rms:.4f} max={amax:.4f}{bad}", flush=True)
        return fn

    if rank == 0:
        print("=== activation trace ===", flush=True)
    dec = module.decoder if hasattr(module, "decoder") else module.module.decoder
    for i, layer in enumerate(dec.layers):
        layer.register_forward_hook(hook(f"layer{i:02d}"))
    if hasattr(dec, "final_layernorm") and dec.final_layernorm is not None:
        dec.final_layernorm.register_forward_hook(hook("final_layernorm"))


def main():
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.training import get_model
    from megatron.core.enums import ModelType

    import miles_plugins.mbridge  # noqa: F401
    from miles.backends.megatron_utils.arguments import set_default_megatron_args
    from miles.backends.megatron_utils.initialize import init
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    # miles' init() calls mpu.initialize_model_parallel, which asserts distributed
    # is already up -- it does not bring it up itself. tools/convert_hf_to_torch_dist.py
    # does this same preamble for the same reason.
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    margs = set_default_megatron_args(parse_args(add_parity_args))

    # These are miles-specific fields that parse_args does not populate, and that
    # miles' init() reads unconditionally. tools/convert_hf_to_torch_dist.py's
    # get_args() sets the same ones for the same reason -- taken as a block rather
    # than one at a time, since discovering them by AttributeError one per run is a
    # slow way to copy a preamble.
    margs.debug_deterministic_collective = False
    margs.enable_witness = False
    margs.save_interval = 1
    margs.micro_batch_size = 1
    margs.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    # The GDN backend flag lives in miles' argument parser, not Megatron's, so
    # this harness sets it directly. flashqla is the fallback for fla's
    # chunk_fwd_o nondeterminism on Blackwell.
    margs.linear_attention_backend = os.environ.get("GDN_BACKEND", "fla")
    if dist.get_rank() == 0 if dist.is_initialized() else True:
        print(f"GDN_BACKEND={margs.linear_attention_backend}", flush=True)

    validate_args(margs)
    init(margs)

    # Apply the dumper's source patches before the model is built, the way sglang's
    # model_runner does. Nothing happens unless DUMPER_SOURCE_PATCHER_CONFIG is set,
    # so this stays inert outside a parity run.
    try:
        from sglang.srt.debug_utils.dumper import dumper

        dumper.apply_source_patches()
    except Exception as exc:  # noqa: BLE001 - debug-only, never fatal
        print(f"source patcher skipped: {exc}", flush=True)

    provider = get_model_provider_func(margs)
    model = get_model(provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    for m in model:
        m.eval()

    if margs.dump:
        _install_dump(model[0])
    if margs.trace:
        _check_key_weights(model[0], margs.hf_checkpoint)
        _check_param_replication(model[0])
        _install_trace(model[0])

    ids = torch.load(margs.tokens).to(torch.long).cuda()
    seq = ids.numel()
    # Megatron wants [b, s]; the model is causal so one sequence is enough.
    input_ids = ids.view(1, seq)
    position_ids = torch.arange(seq, device=ids.device).view(1, seq)
    attention_mask = None

    # The gated-delta-net wrapper asserts packed_seq_params is not None -- it reads
    # cu_seqlens_q for the linear attention -- so these models are always fed packed
    # (THD) batches. One sequence means cu_seqlens = [0, T].
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device=ids.device)
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=seq,
        max_seqlen_kv=seq,
        qkv_format="thd",
    )

    # PLE reads its n-gram row ids off a side channel, because a transformer layer
    # never sees token ids. Nothing is published by default and the PLE forward
    # raises rather than skipping its increment, so this has to be set up here.
    from miles_plugins.models.qwen3_8_next.ops.ple_context import ple_forward_context
    from miles_plugins.models.qwen3_8_next.ops.ple_hash import build_ngram_contexts

    ple = _find_ple_embedding(model[0])
    if ple is None:
        raise RuntimeError(
            "no PLE embedding found in the model; the layer spec should have attached "
            "one to the attention hyper-connection of layer 1"
        )
    contexts = build_ngram_contexts(ids, ple.ngram_size, ple.eos_token_id)
    ngram_ids = ple.compute_ngram_ids(contexts)

    if os.environ.get("PARITY_FIND_NONDETERMINISM") == "1":
        # Every CUDA op without a deterministic implementation raises, naming
        # itself: the fastest way to enumerate what makes two identical forwards
        # disagree by 50%. CUBLAS_WORKSPACE_CONFIG is set by the run script.
        torch.use_deterministic_algorithms(True, warn_only=False)

    # Must exist before the first forward: pass 1 is what records the reference
    # outputs. Creating it between the two forwards records nothing and reports a
    # vacuous "0 of 0".
    if os.environ.get("PARITY_CHUNK_DOUBLE") == "1":
        _wrap_chunk_double_call(model[0])
    if os.environ.get("PARITY_FLA_SUBPROBE") == "1":
        _wrap_fla_subkernels()
    if os.environ.get("PARITY_CHUNK_O_BK"):
        _pin_chunk_o_config()
    bisect = (
        _RepeatBisect(model[0])
        if os.environ.get("PARITY_REPEAT_BISECT") == "1"
        else None
    )

    if margs.backward:
        _run_backward_parity(model, margs, input_ids, position_ids, attention_mask,
                             packed_seq_params, ngram_ids, cu_seqlens, ids)
        return

    with torch.no_grad(), ple_forward_context(ngram_ids, cu_seqlens=cu_seqlens):
        out = model[0](
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )

        # Did the first forward mutate the model? Two identical forwards differing
        # by ~0.7 in the logits is far too large for kernel noise, and the model is
        # in eval() under no_grad, so the suspect is state carried in a parameter or
        # buffer -- which would be a real training bug, not just a parity nuisance.
        snap = {
            n: t.detach().clone()
            for n, t in list(model[0].named_parameters()) + list(model[0].named_buffers())
            if torch.is_tensor(t) and t.device.type == "cuda"
        }

        # Second forward in the same process, compared bitwise against the first.
        # Splits the run-to-run drift into its two possible sources: triton
        # autotune picking different kernel configs per process (in-process
        # repeat identical, cross-run different) versus genuinely nondeterministic
        # kernels (in-process repeat already differs).
        if bisect is not None:
            bisect.pass_no = 2
        out2 = model[0](
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
        if dist.get_rank() == 0:
            changed = []
            live = dict(
                list(model[0].named_parameters()) + list(model[0].named_buffers())
            )
            for n, before in snap.items():
                after = live.get(n)
                if after is None:
                    continue
                d = (after.detach().float() - before.float()).abs().max().item()
                if d > 0:
                    changed.append((n, d))
            print(f"MUTATED_BY_FORWARD count={len(changed)}", flush=True)
            for n, d in changed[:12]:
                print(f"  MUTATED {n:<74} max|d|={d:.6g}", flush=True)

        l1 = (out if isinstance(out, torch.Tensor) else out[0]).float()
        l2 = (out2 if isinstance(out2, torch.Tensor) else out2[0]).float()
        if bisect is not None:
            bisect.report()
        drift = (l1 - l2).abs()
        if dist.get_rank() == 0:
            print(
                f"IN_PROCESS_REPEAT max|d|={drift.max().item():.3e} "
                f"mean|d|={drift.mean().item():.3e} "
                f"identical={'YES' if drift.max().item() == 0 else 'NO'}",
                flush=True,
            )

    if margs.dump:
        from sglang.srt.debug_utils.dumper import dumper

        dumper.step()

    logits = out if isinstance(out, torch.Tensor) else out[0]

    # The model is built with parallel_output, so this is the local vocab shard
    # (248k / TP wide), not the full vocabulary. A plain log_softmax + gather over
    # it is doubly wrong: the normaliser misses the other ranks' logits, and target
    # ids beyond the local shard index out of bounds -- which is how this surfaced,
    # as an async "scatter gather kernel index out of bounds" device assert.
    # miles' own compute_log_probs handles both via fused_vocab_parallel_cross_entropy,
    # and using it means the parity number is produced by the same code path as the
    # RL trainer's logprobs rather than a lookalike.
    from megatron.core import parallel_state
    from miles.backends.training_utils.loss_hub.math_utils import compute_log_probs

    if margs.trace and dist.get_rank() == 0:
        with torch.no_grad():
            lf = logits.detach().float()
            print(f"=== logits: shape={tuple(logits.shape)} rms={lf.pow(2).mean().sqrt():.4f} "
                  f"min={lf.min():.4f} max={lf.max():.4f} "
                  f"per-token std={lf.std(dim=-1).mean():.4f}", flush=True)

    if logits.dim() != 3 or 1 not in logits.shape[:2]:
        raise RuntimeError(f"expected logits with a singleton batch dim, got {tuple(logits.shape)}")
    flat_logits = logits.squeeze(1) if logits.shape[1] == 1 else logits.squeeze(0)

    # Score each token from the position before it, so the numbers line up with what
    # an inference engine reports for the same prompt.
    picked = compute_log_probs(
        flat_logits[:-1].to(torch.float32, copy=True),
        ids[1:],
        parallel_state.get_tensor_model_parallel_group(),
    ).unsqueeze(0)

    if dist.get_rank() == 0:
        os.makedirs(margs.parity_out, exist_ok=True)
        torch.save(
            {"input_ids": ids.cpu(), "logprobs": picked[0].cpu(), "logits_shape": tuple(logits.shape)},
            os.path.join(margs.parity_out, "megatron_logprobs.pt"),
        )
        print(f"MEGATRON_LOGPROBS_OK seq={seq} logits={tuple(logits.shape)}")
    dist.barrier()


if __name__ == "__main__":
    main()

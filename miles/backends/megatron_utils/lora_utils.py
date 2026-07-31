"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.lora import is_lora_enabled, lora_rollout_enabled  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified HF <-> Megatron module name mappings
# ---------------------------------------------------------------------------

# Standard LoRA: merged Q/K/V and merged up/gate
_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
    # GDN (Qwen3.5/Qwen3-Next): both slices live in the single fused megatron in_proj
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
}

_STANDARD_LORA_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate
_CANONICAL_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
}

_CANONICAL_LORA_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

# Megatron -> HF (inverse mapping, one-to-many)
# Covers both standard LoRA (merged) and CanonicalLoRA (split) module names.
_MEGATRON_TO_HF_MODULES = {
    # Standard LoRA (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    # CanonicalLoRA (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    # GDN linear attention: SGLang serves the fused in_proj as two modules
    "in_proj": ["in_proj_qkvz", "in_proj_ba"],
}

_HF_MODULE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkvz",
    "in_proj_ba",
}

# DeepSeek / Kimi MLA (HF names on checkpoint; Megatron uses linear_* from Megatron-Bridge mappings).
_MLA_HF_TO_MEGATRON = {
    "q_a_proj": "linear_q_down_proj",
    "kv_a_proj_with_mqa": "linear_kv_down_proj",
    "q_b_proj": "linear_q_up_proj",
    "kv_b_proj": "linear_kv_up_proj",
    # DSA indexer (GLM-5 / DeepSeek-V3.2): HF/SGLang leaf names vs Megatron-Bridge linear_* names.
    "wq_b": "linear_wq_b",
    "wk": "linear_wk",
    "weights_proj": "linear_weights_proj",
}
_MEGATRON_MLA_TO_HF = {v: k for k, v in _MLA_HF_TO_MEGATRON.items()}

# Empty: dropping a module here makes sglang silently skip its shipped adapter tensors.
_SGLANG_UNSUPPORTED_HF_TARGETS = frozenset()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and getattr(args, "colocate", False) and getattr(args, "lora_base_cpu_backup", False)


_marked_lora_grad_params_cache: dict[int, list] = {}


def reduce_marked_lora_grads(model: Sequence[torch.nn.Module]) -> None:
    """Sum partial grads of replicated native-LoRA params over their tagged group, pre-DP-reduce.

    A native adapter's ``A`` on a column-parallel module is replicated across the TP
    group while each rank only holds its slice of ``B``, so every rank computes a
    partial ``dL/dA`` and the true gradient is their sum. The same applies to a
    replicated ``B`` (row-parallel) or to both sides (MLA's replicated
    down-projections) once sequence parallelism gives each rank a different sequence
    shard. Params are tagged at creation with ``_lora_grad_sum_group``; ``ep`` is
    accepted as a tag for providers whose adapters are expert-parallel, and this is a
    no-op when nothing is tagged.
    """
    from megatron.core import parallel_state as ps

    key = id(model[0]) if model else 0
    marked = _marked_lora_grad_params_cache.get(key)
    if marked is None:
        marked = []
        for chunk in model:
            for param in chunk.parameters():
                group_name = getattr(param, "_lora_grad_sum_group", None)
                if group_name is not None and param.requires_grad:
                    marked.append((param, group_name))
        _marked_lora_grad_params_cache[key] = marked
    if not marked:
        return
    groups = {
        "tp": (ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_world_size()),
        "ep": (ps.get_expert_model_parallel_group(), ps.get_expert_model_parallel_world_size()),
    }
    for group_name in ("tp", "ep"):
        group, size = groups[group_name]
        if size <= 1:
            continue
        grads = []
        for param, g_name in marked:
            if g_name != group_name:
                continue
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads.append(grad)
        for dt in {g.dtype for g in grads}:
            gs = [g for g in grads if g.dtype == dt]
            if len(gs) == 1:
                dist.all_reduce(gs[0], op=dist.ReduceOp.SUM, group=group)
                continue
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            for g, red in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs), strict=False):
                g.copy_(red)


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Check if model has LoRA layers applied."""
    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


def is_lora_weight_name(name: str) -> bool:
    """Check if a weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


def _native_adapter_shard_name(tp_rank: int, pp_rank: int, ep_rank: int) -> str:
    """Per-rank adapter shard filename. EP is only in the name when it actually shards."""
    suffix = f"_ep{ep_rank}" if ep_rank > 0 else ""
    return f"adapter_megatron_tp{tp_rank}_pp{pp_rank}{suffix}.pt"


_param_grad_buffer_patched = False


def patch_param_grad_buffer_for_colocate_mode_lora() -> None:
    """Patch _ParamAndGradBuffer to use disable_param_buffers_cpu_backup=True.

    In colocate mode with offload_train, torch_memory_saver.pause(tag="default")
    offloads default-region GPU memory.  During LoRA training, base weights are
    frozen (requires_grad=False) so DDP only creates buffers for adapter params.

    This patch ensures those buffers are allocated in the "param_buffer" region
    (enable_cpu_backup=False), making them invisible to pause(tag="default") —
    eliminating the need for resume()/pause() around update_weights.

    The patch is idempotent and only takes effect once.
    """
    global _param_grad_buffer_patched
    if _param_grad_buffer_patched:
        return
    _param_grad_buffer_patched = True

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    _original_init = _ParamAndGradBuffer.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["disable_param_buffers_cpu_backup"] = True
        kwargs["disable_grad_buffers_cpu_backup"] = True
        _original_init(self, *args, **kwargs)

    _ParamAndGradBuffer.__init__ = _patched_init
    logger.info("Patched _ParamAndGradBuffer.__init__ for LoRA colocate mode (disable cpu backup)")


# ---------------------------------------------------------------------------
# Module name conversion
# ---------------------------------------------------------------------------


def _get_lora_class_name(lora_type: type | object | None) -> str:
    """Resolve LoRA type to its class name string."""
    if lora_type is None:
        return "CanonicalLoRA"
    if isinstance(lora_type, type):
        return lora_type.__name__
    return type(lora_type).__name__


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    lora_type: type | object | None = None,
) -> list[str]:
    """Convert HuggingFace LoRA target module names to Megatron format.

    HF:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Megatron (LoRA):          linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (CanonicalLoRA): linear_q, linear_k, linear_v, linear_proj,
                              linear_fc1_up, linear_fc1_gate, linear_fc2

    Special values: "all", "all-linear", "all_linear" -> all standard linear modules.
    If input is already in Megatron format, returns as-is.
    """
    class_name = _get_lora_class_name(lora_type)
    is_canonical = class_name == "CanonicalLoRA"

    all_modules = _CANONICAL_LORA_ALL_MODULES if is_canonical else _STANDARD_LORA_ALL_MODULES
    hf_to_megatron = _CANONICAL_LORA_HF_TO_MEGATRON if is_canonical else _STANDARD_LORA_HF_TO_MEGATRON

    # Handle special "all-linear" variants
    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    if isinstance(hf_modules, tuple):
        hf_modules = list(hf_modules)

    # Check if already in Megatron format (standard / canonical / Kimi MLA linear_*).
    if all(m not in _HF_MODULE_NAMES and m not in _MLA_HF_TO_MEGATRON for m in hf_modules if "*" not in m):
        return list(hf_modules)

    # Convert HF names to Megatron names (dedup while preserving order)
    megatron_modules: list[str] = []
    for module in hf_modules:
        if module in _MLA_HF_TO_MEGATRON:
            megatron_name = _MLA_HF_TO_MEGATRON[module]
        else:
            megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)

    return megatron_modules


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron LoRA target module names to HuggingFace format.

    Supports both standard LoRA and CanonicalLoRA module names.

    Megatron standard:   linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron canonical:  linear_q, linear_k, linear_v, linear_proj,
                         linear_fc1_up, linear_fc1_gate, linear_fc2
    HF:                  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Kimi MLA Megatron:   linear_q_down_proj -> q_a_proj, linear_kv_down_proj -> kv_a_proj_with_mqa, ...

    Wildcards (``*.layers.2.mlp.experts.linear_fc1``) get the last dotted
    segment mapped to an HF leaf name; SGLang uses the result to choose
    adapter-buffer types, not to scope by layer.
    """
    if isinstance(megatron_modules, tuple):
        megatron_modules = list(megatron_modules)
    hf_modules: list[str] = []
    for module in megatron_modules:
        lookup_key = module.rsplit(".", 1)[-1] if "." in module else module
        if lookup_key in _MEGATRON_MLA_TO_HF:
            hf_modules.append(_MEGATRON_MLA_TO_HF[lookup_key])
        elif lookup_key in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[lookup_key])
        else:
            # same-name passthrough; SGLang needs the leaf, not a path or pattern
            hf_modules.append(lookup_key)
    seen: set[str] = set()
    unique: list[str] = []
    for m in hf_modules:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def target_modules_hf_for_sglang_rollout(args: Namespace) -> list[str]:
    """HF target_modules for SGLang LoRA init/sync (minus _SGLANG_UNSUPPORTED_HF_TARGETS, currently empty)."""
    raw = list(args.target_modules) if args.target_modules else []
    hf = convert_target_modules_to_hf(raw)
    out = [m for m in hf if m not in _SGLANG_UNSUPPORTED_HF_TARGETS]
    dropped = set(hf) - set(out)
    if dropped:
        logger.warning(
            "target_modules_hf_for_sglang_rollout: omitting %s for SGLang (unsupported by default "
            "get_hidden_dim); Megatron should not train LoRA on these if rollout sync is required.",
            sorted(dropped),
        )
    return out


# ---------------------------------------------------------------------------
# Model setup helpers (used by model.py)
# ---------------------------------------------------------------------------


def parse_exclude_modules(args: Namespace, lora_type=None) -> list[str]:
    """Parse and convert exclude_modules argument."""
    exclude_modules: list[str] = []
    raw = getattr(args, "exclude_modules", None)
    if raw:
        if isinstance(raw, str):
            exclude_modules = [m.strip() for m in raw.split(",")]
        else:
            exclude_modules = list(raw)
        exclude_modules = convert_target_modules_to_megatron(exclude_modules, lora_type=lora_type)
    return exclude_modules


def create_lora_instance(args: Namespace):
    """Create a LoRA or CanonicalLoRA instance based on args.

    Returns:
        A LoRA/CanonicalLoRA dataclass instance ready to be applied to a model.
    """
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_type_name = getattr(args, "lora_type", "lora").lower()

    if lora_type_name == "canonical_lora":
        lora_cls = CanonicalLoRA
    else:
        lora_cls = LoRA

    target_modules = convert_target_modules_to_megatron(args.target_modules, lora_type=lora_cls)
    exclude_modules = parse_exclude_modules(args, lora_type=lora_cls)

    lora_kwargs = dict(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )
    # shared-outer grouped-expert LoRA (SGLang PR #21466); per-expert is the default
    if getattr(args, "experts_shared_outer_loras", False):
        assert lora_cls is LoRA, "--experts-shared-outer-loras requires the standard LoRA adapter type"
        lora_kwargs["experts_shared_outer_loras"] = True

    lora = lora_cls(**lora_kwargs)

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}, "
        f"exclude_modules={exclude_modules}"
    )
    return lora


_DEFAULT_LORA_PROVIDER = "miles_plugins.lora.lora"


def resolve_lora_provider(args: Namespace):
    """Return the module implementing the native-LoRA provider protocol.

    ``--lora-provider-path`` selects a model-specific implementation (a dotted
    module path); the default is the ``miles_plugins.lora`` plugin.
    """
    import importlib

    path = getattr(args, "lora_provider_path", None) or _DEFAULT_LORA_PROVIDER
    module = importlib.import_module(path)
    for entry_point in ("wrap_model_provider_with_lora", "load_lora_adapter_hf", "export_lora_hf_named"):
        assert hasattr(module, entry_point), f"--lora-provider-path {path} must define {entry_point}()"
    return module


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def pp_assemble_full_adapter(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Assemble the complete adapter on every PP rank (the exporter gathers TP/EP, not PP)."""
    import math

    pp_group = get_parallel_state().pp.group
    pp_size = dist.get_world_size(group=pp_group)
    if pp_size == 1:
        return hf_named_tensors
    pp_rank = dist.get_rank(group=pp_group)
    global_ranks = dist.get_process_group_ranks(pp_group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp_size
    dist.all_gather_object(all_meta, local_meta, group=pp_group)

    local_by_name = {n: t for n, t in hf_named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src_pp, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src_pp == pp_rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src_pp], group=pp_group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Save LoRA adapter checkpoint to disk.

    Saves in two formats:
    1. **HF PEFT format** (``adapter_model.safetensors`` + ``adapter_config.json``) for
       external tool compatibility, and for reloading through ``--lora-adapter-path``.
       Bridge mode exports via Megatron-Bridge's ``export_adapter_weights``; raw mode
       via the native provider, whose adapters the bridge exporter cannot see. Both
       handle fused QKV / gate-up splitting and TP gathering. Tensors are cloned before
       writing because that splitting aliases them -- a fused ``linear_fc1`` has one
       ``lora_A`` that exports under both ``gate_proj`` and ``up_proj``, and its ``B``
       becomes two row views -- and ``safetensors`` refuses to write shared storage.
    2. **Megatron-native format** (``adapter_megatron_tp{tp}_pp{pp}_ep{ep}.pt``) for
       fast checkpoint resume without name/weight conversion. Each TP/PP/EP rank saves
       its own shard with original parameter names (ranks sharing ``(tp, pp)`` hold
       different local experts once EP > 1, so the shard key includes the EP rank).

    When ``optimizer`` is provided, training state (optimizer + LR scheduler) is
    also saved per-rank for checkpoint resume. Base model weights are frozen and
    never change, so they are not saved.

    This function is collective: **all ranks must call it** because the bridge
    export performs TP all-gather internally. Only ``dp_rank == 0`` writes files.
    """
    import json

    from megatron.bridge import AutoBridge
    from safetensors.torch import save_file

    from miles.utils import megatron_bridge_utils

    save_path = Path(save_dir)
    parallel_state = get_parallel_state()
    is_dp_cp_rank_0 = parallel_state.effective_dp.rank == 0 and parallel_state.cp.rank == 0
    tp_rank = parallel_state.tp.rank
    pp_rank = parallel_state.pp.rank
    ep_rank = parallel_state.ep.rank

    save_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    if is_dp_cp_rank_0:
        adapter_state: dict[str, torch.Tensor] = {}
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if _is_adapter_param_name(name):
                    adapter_state[name] = param.data.cpu()

        native_path = save_path / _native_adapter_shard_name(tp_rank, pp_rank, ep_rank)
        torch.save(adapter_state, native_path)
        logger.info(f"Saved {len(adapter_state)} adapter tensors (native) to {native_path}")

    lora_state_dict: dict[str, torch.Tensor] = {}
    if getattr(args, "megatron_to_hf_mode", "raw") == "bridge":
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        with megatron_bridge_utils.patch_megatron_model(model):
            for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
                model,
                cpu=True,
                show_progress=False,
            ):
                lora_state_dict[hf_name] = weight
    else:
        for hf_name, weight in resolve_lora_provider(args).export_lora_hf_named(model):
            lora_state_dict[hf_name] = weight.cpu()

    if parallel_state.pp.size > 1:
        assembled = pp_assemble_full_adapter([(name, w.cuda()) for name, w in lora_state_dict.items()])
        lora_state_dict = {name: w.cpu() for name, w in assembled}

    if is_dp_cp_rank_0 and tp_rank == 0 and pp_rank == 0:
        save_file(
            {name: weight.detach().contiguous().clone() for name, weight in lora_state_dict.items()},
            save_path / "adapter_model.safetensors",
        )

        target_modules_hf = (
            convert_target_modules_to_hf(list(args.target_modules))
            if args.target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        config = {
            "peft_type": "LORA",
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": target_modules_hf,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        os.sync()
        logger.info(f"Saved HF PEFT adapter to {save_path} with {len(lora_state_dict)} tensors")

    # ---- Training state (optimizer + scheduler) for resume ----
    if optimizer is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save(
            {
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
            },
            save_path / f"training_state_rank{rank}.pt",
        )
        logger.info(f"Saved optimizer/scheduler state to {save_path}")

    if dist.is_initialized():
        dist.barrier()

    return str(save_path)


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    """Load LoRA adapter weights from a saved checkpoint into the model.

    Attempts to load from Megatron-native format first (per-rank ``.pt`` files),
    which preserves the exact TP/PP sharding and requires no name conversion.
    Falls back to the HF PEFT adapter if native files are not found (not yet
    implemented for HF PEFT format here; ``--lora-adapter-path`` loads that format
    through the native provider's ``load_lora_adapter_hf`` instead).

    When ``optimizer`` is provided, also restores training state (optimizer +
    LR scheduler) from a co-located ``training_state_rank*.pt`` file.

    Args:
        model: List of DDP-wrapped model chunks with LoRA layers already applied.
        adapter_path: Path to the adapter checkpoint directory.
        optimizer: If provided, restore optimizer state for training resume.
        opt_param_scheduler: If provided, restore LR scheduler state.

    Returns:
        ``(loaded, iteration)`` — *loaded* is True if adapter weights were
        successfully loaded; *iteration* is the saved iteration number (or None
        if no training state was found).
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(f"LoRA adapter path does not exist: {adapter_dir}")
        return False, None

    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank
    ep_rank = get_parallel_state().ep.rank

    # ---- Try Megatron-native format first (fast, no conversion needed) ----
    native_path = adapter_dir / _native_adapter_shard_name(tp_rank, pp_rank, ep_rank)
    if native_path.exists():
        state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
        loaded = 0
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(device=param.device))
                    loaded += 1
        logger.info(f"Loaded {loaded} adapter tensors from Megatron-native checkpoint: {native_path}")

        iteration = _load_training_state(adapter_dir, optimizer, opt_param_scheduler)
        return True, iteration

    # ---- HF PEFT format (future work) ----
    hf_path = next(
        (adapter_dir / n for n in ("adapter_model.safetensors", "adapter_model.bin") if (adapter_dir / n).exists()),
        None,
    )
    if hf_path is not None:
        logger.warning(
            f"Found HF PEFT adapter at {hf_path} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_tp*_pp*.pt files) for checkpoint resume."
        )
        return False, None

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False, None


def _load_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    """Restore optimizer/scheduler state saved alongside a LoRA adapter checkpoint."""
    if optimizer is None:
        return None

    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = adapter_dir / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None

    # Optimizer state dicts may contain non-tensor objects (e.g. step counts,
    # param group metadata), so full unpickling is required here.
    training_state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer.load_state_dict(training_state["optimizer"])
    logger.info("Restored optimizer state from LoRA checkpoint")

    if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
        logger.info("Restored LR scheduler state from LoRA checkpoint")

    iteration = training_state.get("iteration")
    if iteration is not None:
        logger.info(f"Resuming LoRA training from iteration {iteration}")
    return iteration


# ---------------------------------------------------------------------------
# LoRA config dict for weight sync to SGLang
# ---------------------------------------------------------------------------


def build_lora_sync_config(args: Namespace) -> dict[str, Any]:
    """Build LoRA config dict for syncing weights to SGLang engines."""
    target_modules_hf = (
        target_modules_hf_for_sglang_rollout(args)
        if args.target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules_hf,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

"""HF/PEFT export and load for native adapters.

Both sides are driven entirely by the adapter's self-description (``kind`` +
``hf_prefix`` + ``shard_meta``) joined against ``_ADAPTER_LAYOUT``, so they know
nothing about models or specs.
"""

from __future__ import annotations

import logging
import os
import re
import time

import torch
import torch.distributed as dist

from .adapter import COLUMN, REPLICATED, ROW, _iter_adapters, _projections

logger = logging.getLogger(__name__)


class _TpGather:
    """Batches per-tensor TP all-gathers into one flat all-gather.

    Requesting returns a handle; ``flush()`` performs the single collective and the
    handles resolve afterwards. Every rank must request in the same order.
    """

    def __init__(self):
        self._requests: list[tuple[torch.Tensor, int]] = []
        self._resolved: list[torch.Tensor] | None = None

    def request(self, local: torch.Tensor, cat_dim: int):
        index = len(self._requests)
        self._requests.append((local, cat_dim))
        return lambda: self._resolved[index]

    def flush(self) -> None:
        from megatron.core import parallel_state as ps

        world = ps.get_tensor_model_parallel_world_size()
        if world == 1 or not self._requests:
            self._resolved = [local for local, _ in self._requests]
            return
        assert len({local.dtype for local, _ in self._requests}) == 1, "mixed adapter dtypes"
        flats = [local.detach().contiguous().reshape(-1) for local, _ in self._requests]
        sizes = [f.numel() for f in flats]
        local_flat = torch.cat(flats)
        gathered = local_flat.new_empty(world * local_flat.numel())
        dist.all_gather_into_tensor(gathered, local_flat, group=ps.get_tensor_model_parallel_group())
        per_rank = gathered.view(world, -1)

        self._resolved = []
        offset = 0
        for (local, cat_dim), size in zip(self._requests, sizes, strict=True):
            shards = [per_rank[r, offset : offset + size].view(local.shape) for r in range(world)]
            self._resolved.append(torch.cat(shards, dim=cat_dim))
            offset += size


def export_lora_hf_named(model_chunks) -> list[tuple[str, torch.Tensor]]:
    """Return ``(hf_name, full_tensor)`` for every adapter param, in HF/PEFT layout.

    TP shards are gathered, so every rank returns the same complete set (e.g.
    ``model.layers.3.self_attn.q_proj.lora_A.weight``). PP is not gathered here;
    callers that need the whole model's adapter assemble across PP stages.

    The logged ``max|lora_B|`` is 0 exactly while the adapter is still at its zero
    init, which separates "the sync works" from "the sync carries anything".
    """
    started = time.perf_counter()
    gather = _TpGather()
    plan: list[tuple[str, object]] = []

    for adapter in _iter_adapters(model_chunks):
        prefix = adapter.hf_prefix
        for proj in _projections(adapter.kind):
            a = getattr(adapter, f"{proj.attr}_A")
            b = getattr(adapter, f"{proj.attr}_B")
            if proj.layout == COLUMN:
                a, b = a, gather.request(b, 0)
            elif proj.layout == ROW:
                a, b = gather.request(a, 1), b
            plan.append((f"{prefix}{proj.hf}.lora_A.weight", a))
            plan.append((f"{prefix}{proj.hf}.lora_B.weight", b))

    gather.flush()
    exported = [
        (name, (source() if callable(source) else source).detach().to(torch.bfloat16).contiguous())
        for name, source in plan
    ]
    if not dist.is_initialized() or dist.get_rank() == 0:
        peak_b = max(
            (t.abs().max().item() for name, t in exported if name.endswith("lora_B.weight")),
            default=0.0,
        )
        logger.info(
            "[lora-native] exported %d adapter tensors in %.1f ms (max|lora_B|=%.3e)",
            len(exported),
            (time.perf_counter() - started) * 1e3,
            peak_b,
        )
    return exported


def load_lora_adapter_hf(model_chunks, adapter_path: str) -> int:
    """Load an HF/PEFT adapter into the attached params, slicing each to this rank.

    Call after the base checkpoint load: adapter params live outside the
    dist-checkpoint, so a base load would not touch (or would clobber) them.
    """
    from safetensors import safe_open

    path = os.path.join(adapter_path, "adapter_model.safetensors")
    assert os.path.exists(path), (
        f"[lora-native] no adapter_model.safetensors under {adapter_path}; "
        "checkpoints written by save_lora_checkpoint use that name"
    )
    loaded = 0
    with safe_open(path, framework="pt") as f:
        keys = {re.sub(r"^base_model\.model\.", "", k): k for k in f.keys()}

        def take(name: str) -> torch.Tensor:
            assert name in keys, f"[lora-native] adapter tensor missing: {name}"
            return f.get_tensor(keys[name])

        def copy_into(param: torch.Tensor, tensor: torch.Tensor) -> None:
            nonlocal loaded
            assert (
                param.shape == tensor.shape
            ), f"[lora-native] shape mismatch: param {tuple(param.shape)} vs adapter slice {tuple(tensor.shape)}"
            with torch.no_grad():
                param.copy_(tensor.to(dtype=param.dtype, device=param.device))
            loaded += 1

        for adapter in _iter_adapters(model_chunks):
            prefix, meta = adapter.hf_prefix, adapter.shard_meta
            tp_rank = meta["tp_rank"]
            for proj in _projections(adapter.kind):
                a_full = take(f"{prefix}{proj.hf}.lora_A.weight")
                b_full = take(f"{prefix}{proj.hf}.lora_B.weight")
                if proj.layout != REPLICATED:
                    width = proj.width(meta)
                    span = slice(tp_rank * width, (tp_rank + 1) * width)
                    if proj.layout == COLUMN:
                        b_full = b_full[span]
                    else:
                        a_full = a_full[:, span]
                copy_into(getattr(adapter, f"{proj.attr}_A"), a_full)
                copy_into(getattr(adapter, f"{proj.attr}_B"), b_full)
    logger.info("[lora-native] loaded %d adapter tensors from %s", loaded, adapter_path)
    return loaded

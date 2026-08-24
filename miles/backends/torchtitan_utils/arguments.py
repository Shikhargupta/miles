"""Arguments for the torchtitan backend.

Extends FSDPArgs rather than restating it: those fields are the common
non-Megatron training options (optimizer, LR schedule, precision, profiling),
not FSDP-specific ones. Only the titan-specific knobs are added here, and
torchtitan's own deeper configuration stays reachable through its config tree
rather than being mirrored into argparse.
"""

import dataclasses
from dataclasses import dataclass

from miles.backends.fsdp_utils.arguments import (
    FSDPArgs,
    build_dataclass_parser,
    load_args_from_parser,
)


@dataclass
class TorchtitanArgs(FSDPArgs):
    # Which torchtitan model to build: resolved as
    # torchtitan.models.<name>.model_registry(<flavor>).
    # Dense flavors only, for now. At the pinned torchtitan commit the qwen3
    # state-dict adapter maps moe.experts.w1/w2/w3 while the model names those
    # parameters w1_EFD/w2_EDF/w3_EFD, and the lookup miss is a silent `continue`
    # -- so an MoE checkpoint's expert weights are never requested and the load
    # fails on 15 unpopulated parameters. Upstream has since restructured this
    # (main uses experts.inner_experts), which arrives with the torch 2.12 bump.
    titan_model_name: str = "qwen3"
    titan_model_flavor: str = "0.6B"

    # "sdpa" | "flex" | "flex_flash" | "varlen". Only sdpa works on torch 2.11
    # (flex/varlen call torch 2.12 APIs), and sdpa is causal-only, so packed
    # multi-document microbatches need the torch bump -- see compat.py.
    titan_attn_backend: str = "sdpa"

    # Sequence length titan sizes its RoPE caches for; must cover the longest
    # packed microbatch (prompt + response).
    titan_seq_len: int = 4096

    # Truncate the built model to the first N transformer blocks (0 = keep all).
    # For loading a few-layer cutdown of a large checkpoint, whose depth has to
    # match exactly. Structural validation only: per-block init scaling was
    # already computed for the full depth, which is harmless because real weights
    # overwrite it, but it makes a from-scratch run with this flag meaningless.
    titan_num_layers: int = 0

    titan_tp_size: int = 1
    titan_pp_size: int = 1
    titan_cp_size: int = 1
    titan_ep_size: int = 1

    wandb_project: str = "miles-torchtitan"


def build_torchtitan_parser(extra_args_provider=None):
    return build_dataclass_parser(TorchtitanArgs, "torchtitan Training (miles)", extra_args_provider)


def load_torchtitan_args(extra_args_provider=None):
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    if args.titan_attn_backend != "sdpa":
        raise ValueError(
            f"--titan-attn-backend {args.titan_attn_backend} needs torch>=2.12; this image pins "
            "torch==2.11.0 (sglang requirement). Use sdpa."
        )
    # sdpa applies a plain causal mask, so a microbatch holding more than one
    # document would let tokens attend across the document boundary.
    if args.use_dynamic_batch_size or args.micro_batch_size != 1:
        raise ValueError(
            "The torchtitan backend currently requires --micro-batch-size 1 and no "
            "--use-dynamic-batch-size: its only torch-2.11-compatible attention backend (sdpa) "
            "is causal-only and cannot mask document boundaries within a packed microbatch."
        )
    if args.titan_pp_size != 1:
        raise ValueError("--titan-pp-size > 1 is not supported yet (the PP schedule owns the microbatch loop)")
    # Mirrors placement_group.py's with_ref. The actor asserts this too, but from
    # inside a Ray actor, where it surfaces as a wrapped RayTaskError after the
    # engines have already started.
    if args.kl_coef != 0 or args.use_kl_loss:
        raise ValueError(
            "The torchtitan backend has no reference model yet, so --use-kl-loss or a non-zero "
            "--kl-coef would train a different objective than the one requested."
        )
    if args.save_debug_train_data is not None:
        raise ValueError("--save-debug-train-data is not wired up for the torchtitan backend")

    known = {f.name for f in dataclasses.fields(TorchtitanArgs)}
    assert "titan_model_name" in known  # guards against a silent rename

"""Build a torchtitan model and load HF weights into it.

Both directions of the HF mapping come from the model's own
``state_dict_adapter``: ``from_hf`` for loading the initial checkpoint and
``to_hf`` for streaming weights to the rollout engines. There is no
miles-specific weight-name table for this backend.
"""

import importlib
import logging
from argparse import Namespace

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

logger = logging.getLogger(__name__)


def titan_state_dict(model: torch.nn.Module) -> dict:
    """The model's state dict under the FQNs torchtitan's adapters expect.

    ``model.state_dict()`` keeps the wrapper prefixes that activation
    checkpointing and FSDP insert (``layers.0._checkpoint_wrapped_module.moe...``),
    and the state-dict adapters match on the unwrapped names. torchtitan reaches
    them through DCP's ModelWrapper for the same reason. Dense models hid this:
    they come back unwrapped, so only an AC-wrapped block (the MoE layers) shows
    the mismatch.
    """
    return get_model_state_dict(model)


# The segments torch inserts when it wraps a module. DCP strips these to build
# the FQNs get_/set_model_state_dict speak; named_parameters() keeps them, so a
# comparison across the two has to strip them the same way.
_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod", "module", "_flat_param"}
)


def _unwrapped_fqn(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def unloaded_parameters(missing_keys, parameter_names) -> list[str]:
    """Which of ``missing_keys`` name real parameters rather than runtime buffers.

    An HF checkpoint carries parameters, not the model's runtime buffers: a
    torchtitan MoE keeps its aux-loss-free load-balancing bias (``expert_bias_E``)
    as a buffer that ``init_weights`` already set up, and no HF export contains
    it. So a missing parameter is a real failure and a missing buffer is expected.

    Both sides are unwrapped rather than assuming which convention
    ``missing_keys`` uses: activation checkpointing inserts
    ``_checkpoint_wrapped_module`` segments, and comparing a wrapped name against
    an unwrapped one matches nothing in either direction, which would report
    unloaded expert weights as skipped buffers and never fail.
    """
    unwrapped = {_unwrapped_fqn(name) for name in parameter_names}
    return [key for key in missing_keys if _unwrapped_fqn(key) in unwrapped]


def resolve_model_spec(args: Namespace):
    """Look up ``model_registry`` in ``torchtitan.models.<name>``.

    Every titan model package exposes the same factory, so a new architecture is
    reachable by name with no code change here.
    """
    module_name = f"torchtitan.models.{args.titan_model_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"--titan-model-name {args.titan_model_name!r} does not resolve to a torchtitan "
            f"model package ({module_name}). Check the pinned torchtitan checkout."
        ) from e

    registry = getattr(module, "model_registry", None)
    if registry is None:
        raise ValueError(f"{module_name} exposes no model_registry(); cannot build a ModelSpec")

    return registry(args.titan_model_flavor, attn_backend=args.titan_attn_backend)


def build_engine_config(args: Namespace, spec):
    """Assemble the torchtitan config tree that the model and components read.

    ForgeEngine.Config is torchtitan's own "trainer as a library" config: the
    same fields Trainer uses, minus the training loop's own knobs. Reusing it
    keeps this backend on a supported surface instead of a private assembly.
    """
    from torchtitan.experiments.forge.engine import ForgeEngine

    config = ForgeEngine.Config(model_spec=spec, hf_assets_path=args.hf_checkpoint)
    config.training.seq_len = args.titan_seq_len
    config.parallelism.tensor_parallel_degree = args.titan_tp_size
    config.parallelism.pipeline_parallel_degree = args.titan_pp_size
    config.parallelism.context_parallel_degree = args.titan_cp_size
    config.parallelism.expert_parallel_degree = args.titan_ep_size
    if args.fp16:
        config.training.dtype = "float16"

    config.optimizer.param_groups = _param_groups(args)
    return config


def _param_groups(args: Namespace) -> list:
    """One catch-all group carrying miles' optimizer settings.

    torchtitan's OptimizersContainer defaults to an empty group list and asserts
    that every trainable parameter is claimed by exactly one group, so a group
    is mandatory rather than optional. Per-group LR/weight-decay splits are a
    torchtitan feature miles does not expose yet; a single ``.*`` group keeps the
    behavior identical to the FSDP backend's single AdamW over all parameters.
    """
    from torchtitan.components.optimizer import ParamGroupConfig

    if args.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.optimizer!r}")

    return [
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs={
                "lr": args.lr,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            },
        )
    ]


def build_model(args: Namespace, spec, config, parallel_dims, device: torch.device):
    """meta-init -> parallelize -> materialize, as in torchtitan's own trainer."""
    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.tools import utils

    model_config = spec.model
    model_config.update_from_config(config=config)

    if args.titan_num_layers:
        available = len(model_config.layers)
        if args.titan_num_layers > available:
            raise ValueError(
                f"--titan-num-layers {args.titan_num_layers} exceeds the "
                f"{args.titan_model_flavor} flavor's {available} blocks"
            )
        model_config.layers = model_config.layers[: args.titan_num_layers]
        logger.info(f"Truncated {args.titan_model_flavor} to {args.titan_num_layers} of {available} blocks")

    with torch.device("meta"), utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
        model = model_config.build()

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "torchtitan pipeline parallelism is not wired into the miles training loop yet "
            "(the PP schedule owns the microbatch loop; see --titan-pp-size)"
        )

    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=config.compile,
        ac_config=config.activation_checkpoint,
        dump_folder=config.dump_folder,
    )
    # Same split torchtitan's own trainer makes: an offloaded model materializes
    # on CPU and takes its buffers on the accelerator, so hardcoding the
    # non-offload branch would put the shards in the wrong place.
    if config.training.enable_cpu_offload:
        init_device, buffer_device = torch.device("cpu"), device
    else:
        init_device, buffer_device = device, None

    model.to_empty(device=init_device)
    with torch.no_grad():
        model.init_weights(buffer_device=buffer_device)
    model.train()
    return model_config, model


def build_ref_model(args: Namespace, spec, config, parallel_dims, device: torch.device):
    """A frozen second copy of the model, for reference log probs.

    Parallelized with torchtitan's own ``enable_cpu_offload`` so the two models
    do not both hold HBM: FSDP2 brings each shard up for the forward and puts it
    back. The flag lives on the training config that ``parallelize_fn`` reads, so
    it is set for this build and restored afterwards rather than copied -- the
    config tree holds the model spec, which is not safe to deep-copy.
    """
    if not args.ref_load:
        raise ValueError("--ref-load is required to build a torchtitan reference model")

    was_offloaded = config.training.enable_cpu_offload
    config.training.enable_cpu_offload = True
    try:
        ref_model_config, ref_model = build_model(args, spec, config, parallel_dims, device)
    finally:
        config.training.enable_cpu_offload = was_offloaded

    load_hf_weights(spec, ref_model_config, ref_model, args.ref_load)
    ref_model.eval()
    ref_model.requires_grad_(False)
    logger.info(f"Built a CPU-offloaded torchtitan reference model from {args.ref_load}")
    return ref_model


def load_hf_weights(spec, model_config, model, hf_checkpoint: str):
    """Load an HF safetensors checkpoint through the model's state-dict adapter.

    Mirrors torchtitan's own ``dcp_load(..., from_hf=True)``: build the HF-keyed
    view of the (already sharded) state dict, let DCP fill it from the
    safetensors shards, then map the names back.
    """
    if spec.state_dict_adapter is None:
        raise ValueError(f"torchtitan model {spec.name!r} has no state_dict_adapter; cannot load an HF checkpoint")

    sd_adapter = spec.state_dict_adapter(model_config, hf_checkpoint)
    hf_state_dict = sd_adapter.to_hf(titan_state_dict(model))
    dcp.load(hf_state_dict, storage_reader=sd_adapter.get_hf_storage_reader(hf_checkpoint, False))
    # get_/set_model_state_dict are a pair: both speak the unwrapped FQNs. Writing
    # back with model.load_state_dict() instead would compare unwrapped keys
    # against the model's wrapped ones and report every wrapped tensor missing.
    result = set_model_state_dict(
        model,
        sd_adapter.from_hf(hf_state_dict),
        options=StateDictOptions(strict=False),
    )

    parameter_names = {name for name, _ in model.named_parameters()}
    unloaded = unloaded_parameters(result.missing_keys, parameter_names)
    if unloaded:
        raise RuntimeError(
            f"HF checkpoint {hf_checkpoint} did not populate {len(unloaded)} parameter(s), e.g. {unloaded[:5]}"
        )
    skipped_buffers = len(result.missing_keys) - len(unloaded)
    if skipped_buffers:
        logger.info(f"{skipped_buffers} runtime buffer(s) kept their initialized values (absent from the HF export)")
    logger.info(
        f"Loaded HF weights from {hf_checkpoint}: {len(hf_state_dict)} tensors requested, "
        f"{len(parameter_names)} model parameters, {len(result.missing_keys)} keys unfilled"
    )
    return sd_adapter

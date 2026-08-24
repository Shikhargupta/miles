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

logger = logging.getLogger(__name__)


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
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    model.train()
    return model_config, model


def load_hf_weights(spec, model_config, model, hf_checkpoint: str):
    """Load an HF safetensors checkpoint through the model's state-dict adapter.

    Mirrors torchtitan's own ``dcp_load(..., from_hf=True)``: build the HF-keyed
    view of the (already sharded) state dict, let DCP fill it from the
    safetensors shards, then map the names back.
    """
    if spec.state_dict_adapter is None:
        raise ValueError(f"torchtitan model {spec.name!r} has no state_dict_adapter; cannot load an HF checkpoint")

    sd_adapter = spec.state_dict_adapter(model_config, hf_checkpoint)
    hf_state_dict = sd_adapter.to_hf(model.state_dict())
    dcp.load(hf_state_dict, storage_reader=sd_adapter.get_hf_storage_reader(hf_checkpoint, False))
    result = model.load_state_dict(sd_adapter.from_hf(hf_state_dict), strict=False)

    if result.missing_keys:
        raise RuntimeError(
            f"HF checkpoint {hf_checkpoint} did not populate {len(result.missing_keys)} parameter(s), "
            f"e.g. {result.missing_keys[:5]}"
        )
    logger.info(f"Loaded HF weights from {hf_checkpoint} ({len(hf_state_dict)} tensors)")
    return sd_adapter

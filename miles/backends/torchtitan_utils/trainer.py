"""torchtitan's Trainer, adopted whole as the training black box.

miles does not assemble torchtitan internals: ``Trainer.__init__`` already owns
the entire build (spec -> config tree -> ParallelDims -> parallelize/pipelining
-> init -> optimizers -> LR -> checkpointer -> loss wiring, including handing
the loss to the PP schedule), and ``forward_backward_step`` already hides the
PP/non-PP split behind one call (the seam torchtitan committed to for
integrators in pytorch/torchtitan#3856). What miles adds here is only the RL
coupling:

* ``build_trainer_config`` -- one ``Trainer.Config`` tree from miles args. The
  config tree is the program: the HF checkpoint load is
  ``checkpoint.initial_load_in_hf`` (the checkpointer resolves weights from
  ``hf_assets_path``), the RL loss is ``config.loss`` (so the trainer wires it
  into the pipeline schedule itself), and the dataloader is an empty stub
  because the RL loop feeds microbatches directly.
* ``MilesRLLoss`` -- a ``BaseLoss`` whose targets are microbatch indices: the
  schedule only transports tensors, so each target names the miles batch the
  real loss closure runs on. One class serves train (loss + log dict) and
  forward-only (log-prob compute) via an armed mode.
* ``MilesRLTrainer`` -- the Trainer subclass. It adds nothing to construction;
  it exposes ``step_runner()`` (the shared loop's per-optimizer-step protocol)
  and forward-only passes, both delegating to the trainer's own
  ``forward_backward_step`` / ``pp_schedule``.
* ``hf_weights`` -- HF-named full tensors for the rollout engines, via the
  family's state-dict adapter (dp/tp gathered, pp broadcast).

Version bridging between released torch and the nightly APIs torchtitan tracks
lives in ``compat`` and is installed before the first torchtitan import.
"""

import inspect
import json
import logging
import os
from argparse import Namespace
from collections.abc import Callable, Iterator

import torch

from miles.backends.torchtitan_utils import compat
from miles.backends.training_utils.torch_native_loop import StepMetrics

logger = logging.getLogger(__name__)


def resolve_model_spec(args: Namespace):
    """The single model entry point: ``torchtitan.models.<name>.model_registry``."""
    import importlib

    compat.install()
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


def build_trainer_config(args: Namespace, *, hf_assets_path: str, lr_total_steps: int, dump_subdir: str):
    """One Trainer.Config tree from miles arguments.

    The parallelism section is filled from miles args with the FSDP degree
    left at -1: torchtitan's own ``ParallelDims.from_config`` infers it, and
    the same dims math sizes the batch fields below.
    """
    if args.optimizer != "adam":
        raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.optimizer!r}")

    compat.install()
    from torchtitan.components.optimizer import ParamGroupConfig
    from torchtitan.trainer import Trainer

    from miles.backends.torchtitan_utils.parallel import parallel_dims_from_config

    config = Trainer.Config()
    config.model_spec = resolve_model_spec(args)
    config.hf_assets_path = hf_assets_path
    config.dump_folder = os.path.join(args.save or "./outputs", "torchtitan", dump_subdir)

    config.parallelism.data_parallel_replicate_degree = args.dp_replicate_size
    config.parallelism.tensor_parallel_degree = args.titan_tp_size
    config.parallelism.pipeline_parallel_degree = args.titan_pp_size
    config.parallelism.context_parallel_degree = args.titan_cp_size
    config.parallelism.expert_parallel_degree = args.titan_ep_size
    config.parallelism.pipeline_parallel_microbatch_size = 1
    parallel_dims = parallel_dims_from_config(config.parallelism)
    dp_size = parallel_dims.dp_replicate * parallel_dims.dp_shard

    config.training.seq_len = args.titan_seq_len
    # One miles microbatch (a packed (1, seq) document batch) is one trainer
    # "sample": local_batch_size is the per-rank microbatch count of one
    # optimizer step, so the PP schedule (built from it) matches the RL loop.
    config.training.local_batch_size = max(args.global_batch_size // dp_size // args.micro_batch_size, 1)
    config.training.global_batch_size = config.training.local_batch_size * dp_size
    config.training.steps = max(lr_total_steps, 1)
    config.training.disable_cuda_graphs = True  # microbatch shapes vary across rollouts
    if args.fp16:
        config.training.dtype = "float16"

    # One catch-all group: OptimizersContainer asserts every trainable param is
    # claimed by exactly one group.
    config.optimizer.param_groups = [
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
    config.training.max_norm = args.clip_grad

    config.loss = _miles_loss_cls().Config()
    config.dataloader = _empty_dataloader_cls().Config()
    config.checkpoint = _checkpoint_manager_cls().Config()
    config.activation_checkpoint = None  # parallelize_fn treats None as AC off
    config.debug.seed = args.seed

    # The checkpointer must be enabled for the initial load: with no native
    # checkpoint under dump_folder it falls through to the HF assets load
    # (from_hf via the family's state-dict adapter).
    config.checkpoint.enable = True
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.initial_load_in_hf = True

    # miles owns experiment tracking; titan's metrics stay console-only.
    config.metrics.enable_tensorboard = False
    config.metrics.enable_wandb = False
    config.validator.enable = False
    return config


def _lazy_titan():
    """Import-once holder for the torchtitan base classes.

    Deferred so that importing this module (e.g. by the actor or unit tests)
    does not require torchtitan; compat shims install before first use.
    """
    compat.install()
    from torchtitan.components.dataloader import BaseDataLoader
    from torchtitan.components.loss import BaseLoss
    from torchtitan.trainer import Trainer

    return Trainer, BaseLoss, BaseDataLoader


_TITAN = None


def _titan():
    global _TITAN
    if _TITAN is None:
        _TITAN = _lazy_titan()
    return _TITAN


def make_miles_trainer(config):
    """Instantiate the MilesRLTrainer subclass (created against the imported Trainer)."""
    return _miles_trainer_cls()(config)


# ---------------------------------------------------------------------------
# The classes below reference torchtitan base classes, so they are created
# lazily on first use and cached.

_CLASSES: dict = {}


def _miles_loss_cls():
    if "loss" in _CLASSES:
        return _CLASSES["loss"]
    _, BaseLoss, _ = _titan()
    from dataclasses import dataclass

    class MilesRLLossImpl(BaseLoss):
        """Trampoline between the schedule's (pred, target) and miles' RL loss.

        Targets are microbatch-index tensors: the schedule only transports
        tensors, and the RL loss needs the whole miles batch (advantages, old
        log probs, masks), which stays outside torchtitan. ``arm`` sets the
        batches and closure for the next step; in eval mode the closure result
        is stashed and a zero scalar returned (the schedule requires a loss).

        Results are keyed by microbatch index rather than appended: the
        schedule may invoke the loss extra times outside the scheduled
        microbatches (its first step runs a backward-metadata inference call),
        which upstream's pure losses never notice. Keying makes those calls
        idempotent -- the scheduled pass overwrites, and exactly one result
        per microbatch survives.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(BaseLoss.Config):
            pass

        def __init__(self, config, *, compile_config=None):
            self.config = config
            self._batches: list | None = None
            self._closure: Callable | None = None
            self._mode = "train"
            self._results: dict[int, object] = {}

        def arm(self, batches: list, closure: Callable, mode: str) -> None:
            self._batches, self._closure, self._mode = batches, closure, mode
            self._results = {}

        def collect(self) -> list:
            missing = [i for i in range(len(self._batches)) if i not in self._results]
            if missing:
                raise RuntimeError(f"the schedule never ran microbatch(es) {missing}")
            return [self._results[i] for i in range(len(self._batches))]

        def __call__(self, pred, target, global_valid_tokens=None, **kwargs):
            from torch.distributed.tensor import DTensor

            if isinstance(pred, DTensor):
                # Under TP the lm_head output is Shard(-1); miles' loss code
                # wants plain full-vocab tensors.
                pred = pred.full_tensor()
            index = int(target)
            batch = self._batches[index]
            if self._mode == "train":
                loss, log_dict = self._closure(pred, batch)
                self._results[index] = log_dict
                return loss, {}
            self._results[index] = self._closure(pred, batch)
            return torch.zeros((), device=pred.device, dtype=torch.float32), {}

    _CLASSES["loss"] = MilesRLLossImpl
    return MilesRLLossImpl


def _empty_dataloader_cls():
    if "dataloader" in _CLASSES:
        return _CLASSES["dataloader"]
    _, _, BaseDataLoader = _titan()
    from dataclasses import dataclass

    class EmptyDataLoaderImpl(BaseDataLoader):
        """The RL loop feeds microbatches directly; the trainer's own
        dataloader is never iterated and checkpoints no state."""

        @dataclass(kw_only=True, slots=True)
        class Config(BaseDataLoader.Config):
            pass

        def __init__(self, config, **kwargs):
            self.config = config

        def __iter__(self):
            return iter(())

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

    _CLASSES["dataloader"] = EmptyDataLoaderImpl
    return EmptyDataLoaderImpl


def _checkpoint_manager_cls():
    if "checkpoint" in _CLASSES:
        return _CLASSES["checkpoint"]
    from dataclasses import dataclass

    from torchtitan.components import checkpoint as titan_checkpoint

    class TiedCheckpointManager(titan_checkpoint.CheckpointManager):
        """CheckpointManager whose HF load survives tied checkpoints.

        torchtitan flavors qwen3_5 with a separate ``lm_head`` while the HF
        checkpoint ties it to the embedding and ships no ``lm_head.weight``;
        upstream ``dcp_load`` requests every exported key and dies on the
        missing one. The from_hf branch below is upstream's, plus: keys the
        checkpoint does not ship are dropped from the request (the adapter's
        ``from_hf`` reconstructs them), and when the dropped key is the tied
        lm_head on a rank that owns no embedding (a PP last stage), the
        checkpoint's embedding is requested into an lm_head-shaped skeleton so
        the reconstruction has a source.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(titan_checkpoint.CheckpointManager.Config):
            pass

        def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
            if not from_hf:
                return super().dcp_load(state_dict, checkpoint_id, from_hf, from_quantized)

            assert self.sd_adapter is not None
            hf_state = self.sd_adapter.to_hf(state_dict)
            index_mapping = getattr(self.sd_adapter, "fqn_to_index_mapping", None)
            if index_mapping:
                available = set(index_mapping)
                dropped = sorted(k for k in hf_state if k not in available)
                if dropped:
                    logger.info(
                        f"HF checkpoint lacks {len(dropped)} exported key(s) (e.g. {dropped[:3]}); "
                        "deferring to the adapter's from_hf reconstruction"
                    )
                    lm_head_skeleton = hf_state.get("lm_head.weight")
                    hf_state = {k: v for k, v in hf_state.items() if k in available}
                    if "lm_head.weight" in dropped and lm_head_skeleton is not None:
                        embed_key = next((k for k in available if k.endswith("embed_tokens.weight")), None)
                        if embed_key is not None and embed_key not in hf_state:
                            hf_state[embed_key] = torch.empty_like(lm_head_skeleton)

            titan_checkpoint.dcp.load(
                hf_state,
                storage_reader=self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized),
            )
            self.states[titan_checkpoint.MODEL].load_state_dict(self.sd_adapter.from_hf(hf_state))

    _CLASSES["checkpoint"] = TiedCheckpointManager
    return TiedCheckpointManager


def _miles_trainer_cls():
    if "trainer" in _CLASSES:
        return _CLASSES["trainer"]
    Trainer, _, _ = _titan()

    class MilesRLTrainerImpl(Trainer):
        """torchtitan's Trainer with the RL step surface bolted on.

        Construction is entirely the base class. The additions translate the
        shared RL loop's step-runner protocol onto the trainer's own
        machinery: ``forward_backward_step`` (which internally dispatches PP
        schedule vs single model), the optimizer/LR containers, and titan's
        grad clipping.
        """

        # ------------------------------------------------------------- data

        def _family_forward_kwargs(self) -> dict:
            """Static per-family forward kwargs (resolved once).

            qwen3_5 dereferences ``special_tokens`` unconditionally, text-only
            included; the ids live in the HF config. These ride input_dict:
            ``post_dataloading_process`` forwards every non-"input" key to the
            model, PP stages included.
            """
            if not hasattr(self, "_family_kwargs"):
                self._family_kwargs = {}
                if "special_tokens" in inspect.signature(self.model_parts[0].forward).parameters:
                    hf_cfg = json.load(open(os.path.join(self.config.hf_assets_path, "config.json")))
                    self._family_kwargs["special_tokens"] = {
                        "image_id": hf_cfg.get("image_token_id", -1),
                        "video_id": hf_cfg.get("video_token_id", -2),
                    }
            return self._family_kwargs

        def _rl_input_dict(self, batch: dict) -> dict:
            return {
                "input": batch["tokens"],
                "positions": batch["position_ids"],
                **self._family_forward_kwargs(),
            }

        def _rl_microbatches(self, batches) -> tuple[list[dict], list[torch.Tensor]]:
            batches = list(batches)
            if self.parallel_dims.pp_enabled:
                expected = self.num_pipeline_parallel_microbatches
                if len(batches) != expected:
                    raise ValueError(
                        f"the PP schedule was built for {expected} microbatches per optimizer step "
                        f"but this step has {len(batches)}; global_batch_size / dp / "
                        "micro_batch_size must be constant (no dynamic batch sizing with PP)"
                    )
            input_dicts = [self._rl_input_dict(batch) for batch in batches]
            # Targets are index tensors; MilesRLLoss maps them back to batches.
            labels = [torch.tensor(i, device=self.device) for i in range(len(batches))]
            return input_dicts, labels

        # ---------------------------------------------------- RL step surface

        def rl_forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
            """One optimizer step's microbatches through the trainer's own
            forward_backward_step. Under PP only the last stage returns log
            dicts."""
            batches = list(batches)
            self.loss_fn.arm(batches, loss_closure, "train")
            input_dicts, labels = self._rl_microbatches(batches)
            ones = torch.ones((), device=self.device)
            if self.parallel_dims.pp_enabled:
                self.forward_backward_step(input_dict=input_dicts, labels=labels, global_valid_tokens=ones)
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    self.forward_backward_step(input_dict=input_dict, labels=label, global_valid_tokens=ones)
            return self.loss_fn.collect() if self.pp_last_stage_here() else []

        def rl_forward_only_step(self, batches, compute: Callable) -> list:
            """Forward-only over the microbatches (log probs); mirrors the
            validator's eval path. Under PP only the last stage returns."""
            batches = list(batches)
            self.loss_fn.arm(batches, compute, "eval")
            input_dicts, labels = self._rl_microbatches(batches)
            if self.parallel_dims.pp_enabled:
                arg_mbs, kwarg_mbs, target_mbs = [], [], []
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    arg_mbs.append((inputs,))
                    kwarg_mbs.append(extra)
                    target_mbs.append(label)
                losses = [] if self.pp_has_last_stage else None
                self.pp_schedule.eval(
                    arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                    kwarg_mbs=kwarg_mbs,
                    target_mbs=target_mbs if self.pp_has_last_stage else None,
                    losses=losses,
                )
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    pred = self.model_parts[0](inputs, **extra)
                    self.loss_fn(pred, label)
            return self.loss_fn.collect() if self.pp_last_stage_here() else []

        def rl_zero_grad(self) -> None:
            self.optimizers.zero_grad(set_to_none=True)

        def rl_apply_step(self) -> StepMetrics:
            """The optim block of the trainer's train_step, returning what the
            miles loop logs."""
            from torchtitan.distributed import utils as dist_utils

            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()
            self.step += 1  # the trainer's own step counter, checkpointed as train_state
            return StepMetrics(
                grad_norm=float(grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else grad_norm.item()),
                extra_metrics=self.lr_schedulers.get_metrics(),
            )

        def pp_last_stage_here(self) -> bool:
            return (not self.parallel_dims.pp_enabled) or self.pp_has_last_stage

        def step_runner(self) -> "TrainerStepRunner":
            return TrainerStepRunner(self)

        # ------------------------------------------------------------ weights

        def hf_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
            """HF-named tensors, materialized one at a time, for the engine push.

            The weight transport requires every rank in an IPC gather group to
            stream the same tensor sequence. dp/tp shards reassemble via
            ``gather_full_param``; under PP each tensor lives on exactly one
            stage, so it is additionally broadcast over the pp mesh -- after
            which every rank yields the identical full stream and the transport
            stays PP-oblivious. One tensor is resident at a time either way.
            """
            import torch.distributed as dist

            from miles.backends.fsdp_utils.dtensor import gather_full_param

            sd_adapter = self.checkpointer.sd_adapter
            local = sd_adapter.to_hf({k: v for part in self.model_parts for k, v in part.state_dict().items()})
            if not self.parallel_dims.pp_enabled:
                for name, tensor in local.items():
                    yield name, gather_full_param(tensor)
                return

            pp_group = self.parallel_dims.get_mesh("pp").get_group()
            my_index = dist.get_rank(pp_group)
            # DTensor.shape is the global shape, so the metadata already
            # describes the post-gather tensor.
            local_meta = {name: (my_index, tuple(t.shape), t.dtype) for name, t in local.items()}
            gathered: list = [None] * dist.get_world_size(pp_group)
            dist.all_gather_object(gathered, local_meta, group=pp_group)
            merged: dict = {}
            for meta in gathered:
                for name, entry in meta.items():
                    if name in merged:
                        raise RuntimeError(f"{name} is owned by two pipeline stages")
                    merged[name] = entry

            for name in sorted(merged):
                owner, shape, dtype = merged[name]
                if owner == my_index:
                    tensor = gather_full_param(local[name])
                else:
                    tensor = torch.empty(shape, dtype=dtype, device=self.device)
                dist.broadcast(tensor, src=dist.get_global_rank(pp_group, owner), group=pp_group)
                yield name, tensor

    _CLASSES["trainer"] = MilesRLTrainerImpl
    return MilesRLTrainerImpl


class TrainerStepRunner:
    """Adapter from the shared loop's step-runner protocol to the trainer.

    Kept separate because the trainer already has a ``forward_backward_step``
    with torchtitan's own signature; the protocol must not shadow it.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def forward_only_step(self, batches, compute: Callable) -> list:
        return self.trainer.rl_forward_only_step(batches, compute)

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        return self.trainer.rl_forward_backward_step(batches, loss_closure)

    def zero_grad(self) -> None:
        self.trainer.rl_zero_grad()

    def apply_step(self) -> StepMetrics:
        return self.trainer.rl_apply_step()

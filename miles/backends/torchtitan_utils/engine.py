"""torchtitan behind one boundary.

miles treats torchtitan as a black-box model provider: this module is the only
place in the backend that imports torchtitan or knows its conventions. The
actor and the shared RL loop see an engine with a handful of methods; whether a
run is pipelined, how HF checkpoints map onto titan modules, and which quirks a
family has (fused QKV state-dict hooks, tied-embedding checkpoints, qwen3_5's
mandatory ``special_tokens``) are all internal here.

The step surface mirrors the shape torchtitan's maintainers committed to for
integrators (torchtitan#3856): a single ``forward_backward_step`` per optimizer
step, with the PP schedule and the linear path as two implementations behind
it, and microbatches pre-split by the caller. The caller never conditions on
PP.

Conventions this module encodes (each verified on torchtitan @2f10a2590 with
torch 2.13):

* State dicts use plain ``model.state_dict()`` / ``load_state_dict`` — titan's
  own ModelWrapper does the same. Hook-produced aliases (FusedQKVLinear
  presents ``wq/wk/wv`` for a fused ``wqkv``) only exist on this path; DCP's
  ``get_model_state_dict`` walks module attributes and dies on them.
* The HF-load skeleton is filtered to keys the checkpoint actually ships. A
  tied checkpoint has no ``lm_head.weight``; the adapter's ``from_hf``
  reconstructs it from the embedding, and the fail-closed check after load
  still guards everything else.
* A missing key after load is acceptable only if it names a known runtime
  buffer (MoE ``expert_bias_E``). Comparing against ``named_parameters()``
  instead would silently pass fused modules.
* Under TP the lm_head output is hardcoded ``Shard(-1)``; the engine gathers
  logits to full vocab so miles' loss code stays DTensor-free.
* Grad clipping goes through titan's ``clip_grad_norm_``, which reduces across
  PP stages and handles EP's split meshes.
"""

import importlib
import json
import logging
import os
from argparse import Namespace
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import torch
from torch.distributed.checkpoint.stateful import Stateful

from miles.backends.torchtitan_utils import compat
from miles.backends.torchtitan_utils.parallel import build_parallel_dims, create_titan_parallel_state
from miles.backends.training_utils.torch_native_loop import LinearStepRunner, StepMetrics

logger = logging.getLogger(__name__)

# The segments torch inserts when it wraps a module (activation checkpointing,
# FSDP, compile). Canonical state-dict keys omit them; named_buffers() keeps
# them, so comparisons across the two must strip them the same way.
_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod", "module", "_flat_param"}
)


def _unwrapped_fqn(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def unloaded_parameters(missing_keys, buffer_names) -> list[str]:
    """Which of ``missing_keys`` are real failures rather than runtime buffers.

    Fail closed: a missing key passes only if it names a known buffer. An HF
    checkpoint carries parameters, not runtime buffers (a torchtitan MoE keeps
    its load-balancing bias ``expert_bias_E`` as a buffer that ``init_states``
    already set up), so a missing buffer is expected and anything else is not.
    """
    buffers = {_unwrapped_fqn(name) for name in buffer_names}
    return [key for key in missing_keys if _unwrapped_fqn(key) not in buffers]


class TitanEngine:
    """One torchtitan model, ready for the miles RL loop.

    Construction follows the assembly order torchtitan's own trainer uses:
    spec -> config -> ParallelDims -> update_from_config -> meta build ->
    (pipelining_fn | parallelize_fn) -> to_empty -> init_states -> optimizer
    strictly after parallelisms -> post-optimizer hooks -> LR schedulers.
    """

    def __init__(self, args: Namespace, device: torch.device, *, lr_total_steps: int, parallel_dims=None):
        compat.install()  # no-op wherever torch already has the symbols

        self.args = args
        self.device = device
        self.spec = self._resolve_spec(args)
        # The actor builds ParallelDims early (miles' ParallelState comes from
        # it before any model exists); accepting it avoids a second set of
        # process groups over the same ranks.
        self.parallel_dims = parallel_dims if parallel_dims is not None else build_parallel_dims(args)
        self.config = self._build_config(args)

        self.model_config = self.spec.model
        self.model_config.update_from_config(config=self.config)
        self._truncate_layers(args)

        self.model_parts, self.pp_schedule, self.pp_has_first_stage, self.pp_has_last_stage = (
            self._build_parts(cpu_offload=False)
        )

        self.optimizers = self.config.optimizer.build(model_parts=self.model_parts)
        if self.spec.post_optimizer_build_fn is not None:
            self.spec.post_optimizer_build_fn(self.optimizers, self.model_parts, self.parallel_dims)
        self.lr_schedulers = self.config.lr_scheduler.build(
            optimizers=self.optimizers, training_steps=max(lr_total_steps, 1)
        )

        self._fwd_static_kwargs = self._family_forward_kwargs(args)
        # PP trampoline state: set per forward_backward_step/forward_only_step call.
        self._pp_batches: list | None = None
        self._pp_closure: Callable | None = None
        self._pp_results: list | None = None
        self._pp_mode: str = "train"

    # ------------------------------------------------------------------ build

    @staticmethod
    def _resolve_spec(args: Namespace):
        """The single model entry point: ``torchtitan.models.<name>.model_registry``."""
        module_name = f"torchtitan.models.{args.titan_model_name}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"--titan-model-name {args.titan_model_name!r} does not resolve to a torchtitan "
                f"model package ({module_name}). Check the vendored torchtitan checkout."
            ) from e
        registry = getattr(module, "model_registry", None)
        if registry is None:
            raise ValueError(f"{module_name} exposes no model_registry(); cannot build a ModelSpec")
        return registry(args.titan_model_flavor, attn_backend=args.titan_attn_backend)

    @staticmethod
    def _build_config(args: Namespace):
        """An attribute bag over core torchtitan config types.

        torchtitan's contracts want attribute access on the root and real core
        types as values: ``update_from_config`` isinstance-checks
        ``config.parallelism``; ``parallelize_fn``/``pipelining_fn`` take the
        sections as kwargs; the containers build from their own Configs.
        ``activation_checkpoint=None`` means AC off. Nothing under
        ``torchtitan/experiments`` is used.
        """
        from torchtitan.components.lr_scheduler import LRSchedulersContainer
        from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
        from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig

        if args.optimizer != "adam":
            raise ValueError(f"torchtitan backend supports --optimizer adam, got {args.optimizer!r}")

        config = SimpleNamespace(
            training=TrainingConfig(),
            parallelism=ParallelismConfig(),
            optimizer=OptimizersContainer.Config(
                # One catch-all group: OptimizersContainer asserts every
                # trainable param is claimed by exactly one group.
                param_groups=[
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
            ),
            lr_scheduler=LRSchedulersContainer.Config(),
            compile=CompileConfig(),
            activation_checkpoint=None,
            dump_folder="./outputs",
        )
        config.training.seq_len = args.titan_seq_len
        config.parallelism.tensor_parallel_degree = args.titan_tp_size
        config.parallelism.pipeline_parallel_degree = args.titan_pp_size
        config.parallelism.context_parallel_degree = args.titan_cp_size
        config.parallelism.expert_parallel_degree = args.titan_ep_size
        if args.fp16:
            config.training.dtype = "float16"
        return config

    def _truncate_layers(self, args: Namespace) -> None:
        if not getattr(args, "titan_num_layers", 0):
            return
        available = len(self.model_config.layers)
        if args.titan_num_layers > available:
            raise ValueError(
                f"--titan-num-layers {args.titan_num_layers} exceeds the "
                f"{args.titan_model_flavor} flavor's {available} blocks"
            )
        self.model_config.layers = self.model_config.layers[: args.titan_num_layers]
        logger.info(f"Truncated {args.titan_model_flavor} to {args.titan_num_layers} of {available} blocks")

    def _build_parts(self, *, cpu_offload: bool):
        """meta build -> (pipelining | parallelize) -> materialize -> init."""
        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.tools import utils as titan_utils

        was_offloaded = self.config.training.enable_cpu_offload
        self.config.training.enable_cpu_offload = cpu_offload
        try:
            with (
                torch.device("meta"),
                titan_utils.set_default_dtype(TORCH_DTYPE_MAP[self.config.training.dtype]),
            ):
                model = self.model_config.build()

            if self.parallel_dims.pp_enabled:
                schedule, parts, has_first, has_last = self.spec.pipelining_fn(
                    model,
                    parallel_dims=self.parallel_dims,
                    training=self.config.training,
                    parallelism=self.config.parallelism,
                    compile_config=self.config.compile,
                    ac_config=self.config.activation_checkpoint,
                    dump_folder=self.config.dump_folder,
                    device=self.device,
                    model_config=self.model_config,
                    parallelize_fn=self.spec.parallelize_fn,
                    loss_fn=self._pp_loss_trampoline,
                )
                del model
            else:
                parts = [
                    self.spec.parallelize_fn(
                        model,
                        parallel_dims=self.parallel_dims,
                        training=self.config.training,
                        parallelism=self.config.parallelism,
                        compile_config=self.config.compile,
                        ac_config=self.config.activation_checkpoint,
                        dump_folder=self.config.dump_folder,
                    )
                ]
                schedule, has_first, has_last = None, True, True

            # torchtitan's trainer split: an offloaded model materializes on CPU
            # and takes its buffers on the accelerator.
            init_device = torch.device("cpu") if cpu_offload else self.device
            buffer_device = self.device if cpu_offload else None
            for part in parts:
                part.to_empty(device=init_device)
                with torch.no_grad():
                    part.init_states(buffer_device=buffer_device)
                part.train()
            return parts, schedule, has_first, has_last
        finally:
            self.config.training.enable_cpu_offload = was_offloaded

    def _family_forward_kwargs(self, args: Namespace) -> dict:
        """Static per-family forward kwargs, resolved once.

        qwen3_5 dereferences ``special_tokens`` unconditionally, text-only
        included; the ids live in the HF config. Families whose forward does
        not take the kwarg get nothing.
        """
        import inspect

        static: dict = {}
        if "special_tokens" in inspect.signature(self.model_parts[0].forward).parameters:
            hf_cfg = json.load(open(os.path.join(args.hf_checkpoint, "config.json")))
            static["special_tokens"] = {
                "image_id": hf_cfg.get("image_token_id", -1),
                "video_id": hf_cfg.get("video_token_id", -2),
            }
        return static

    # ------------------------------------------------------------ RL loop seam

    def _forward_kwargs(self, batch: dict) -> dict:
        positions = batch["position_ids"]
        kwargs = {"positions": positions, **self._fwd_static_kwargs}
        model = self.model_parts[0]
        if hasattr(model, "get_attention_masks"):
            kwargs["attention_masks"] = model.get_attention_masks(positions=positions)
        return kwargs

    def _forward(self, batch: dict, module: torch.nn.Module | None = None) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        model = module if module is not None else self.model_parts[0]
        out = model(batch["tokens"], **self._forward_kwargs(batch))
        logits = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(logits, DTensor):
            # Under TP the lm_head output is Shard(-1); miles' loss code wants
            # plain full-vocab tensors.
            logits = logits.full_tensor()
        return logits

    def forward_only_step(self, batches, compute: Callable) -> list:
        """Run ``compute(logits, batch)`` for each microbatch, without grad.

        Under PP only the last stage produces results; other stages return [].
        """
        if not self.parallel_dims.pp_enabled:
            return LinearStepRunner(self._forward).forward_only_step(batches, compute)

        batches = list(batches)  # the schedule needs every microbatch up front
        self._pp_mode, self._pp_batches, self._pp_closure, self._pp_results = (
            "eval", batches, compute, []
        )
        arg_mbs, kwarg_mbs, target_mbs, losses = self._pp_microbatches(batches)
        self.pp_schedule.eval(
            arg_mbs=arg_mbs if self.pp_has_first_stage else None,
            kwarg_mbs=kwarg_mbs,
            target_mbs=target_mbs,
            losses=losses,
        )
        return self._pp_results if self.pp_has_last_stage else []

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        """One optimizer step's worth of microbatches, forward+backward.

        ``loss_closure(logits, batch) -> (loss, log_dict)``. The linear path
        and the PP schedule are the two implementations behind this seam; the
        caller never conditions on PP. Under PP only the last stage returns
        log dicts.
        """
        if not self.parallel_dims.pp_enabled:
            return LinearStepRunner(self._forward).forward_backward_step(batches, loss_closure)

        batches = list(batches)  # the schedule needs every microbatch up front
        self._pp_mode, self._pp_batches, self._pp_closure, self._pp_results = (
            "train", batches, loss_closure, []
        )
        arg_mbs, kwarg_mbs, target_mbs, losses = self._pp_microbatches(batches)
        self.pp_schedule.step(
            arg_mbs=arg_mbs if self.pp_has_first_stage else None,
            kwarg_mbs=kwarg_mbs,
            target_mbs=target_mbs,
            losses=losses,
            return_outputs=False,
        )
        return self._pp_results if self.pp_has_last_stage else []

    def _pp_microbatches(self, batches: list[dict]):
        arg_mbs = [(batch["tokens"],) for batch in batches]
        kwarg_mbs = [self._forward_kwargs(batch) for batch in batches]
        # The schedule's targets must be tensors; each carries only its
        # microbatch index — the trampoline looks the real batch up by it.
        target_mbs = (
            [torch.tensor(i, device=self.device) for i in range(len(batches))]
            if self.pp_has_last_stage
            else None
        )
        losses = [] if self.pp_has_last_stage else None
        return arg_mbs, kwarg_mbs, target_mbs, losses

    def _pp_loss_trampoline(self, pred, target, **_):
        """The loss_fn handed to the schedule at build time.

        The schedule calls it with last-stage logits and the index tensor; the
        real per-microbatch loss (or log-prob compute) runs on the batch the
        index names. Registered once, dispatches on the current mode.
        """
        from torch.distributed.tensor import DTensor

        if isinstance(pred, DTensor):
            pred = pred.full_tensor()
        batch = self._pp_batches[int(target)]
        if self._pp_mode == "train":
            loss, log_dict = self._pp_closure(pred, batch)
            self._pp_results.append(log_dict)
            return loss
        self._pp_results.append(self._pp_closure(pred, batch))
        return torch.zeros((), device=pred.device, dtype=torch.float32)

    def zero_grad(self) -> None:
        self.optimizers.zero_grad(set_to_none=True)

    def apply_step(self) -> StepMetrics:
        from torchtitan.distributed.utils import clip_grad_norm_

        grad_norm = clip_grad_norm_(
            [p for part in self.model_parts for p in part.parameters()],
            self.args.clip_grad,
            foreach=True,
            pp_mesh=(self.parallel_dims.get_mesh("pp") if self.parallel_dims.pp_enabled else None),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        self.optimizers.step()
        self.lr_schedulers.step()
        return StepMetrics(grad_norm=float(grad_norm.item()), extra_metrics=self.lr_schedulers.get_metrics())

    # ---------------------------------------------------------------- weights

    def _local_state_dict(self) -> dict:
        return {k: v for part in self.model_parts for k, v in part.state_dict().items()}

    def load_hf(self, path: str, parts: list | None = None) -> None:
        """Load an HF safetensors checkpoint through the family's adapter."""
        import torch.distributed.checkpoint as dcp

        parts = parts if parts is not None else self.model_parts
        sd_adapter = self.spec.state_dict_adapter(self.model_config, path)
        local_state = {k: v for part in parts for k, v in part.state_dict().items()}
        hf_state = sd_adapter.to_hf(local_state)

        if sd_adapter.fqn_to_index_mapping:
            available = set(sd_adapter.fqn_to_index_mapping)
            dropped = sorted(k for k in hf_state if k not in available)
            if dropped:
                logger.info(
                    f"HF checkpoint lacks {len(dropped)} key(s) the model exports "
                    f"(e.g. {dropped[:3]}); deferring to from_hf reconstruction"
                )
                lm_head_skeleton = hf_state.get("lm_head.weight")
                hf_state = {k: v for k, v in hf_state.items() if k in available}
                # Tied checkpoint x PP: a last stage owns lm_head but not the
                # embedding ``from_hf`` rebuilds it from. Request the
                # checkpoint's embedding into an lm_head-shaped skeleton (same
                # global shape, tied) so the rebuild has a source on this rank;
                # the resulting extra embedding key is ignored by the
                # non-strict load on a stage that has no embedding module.
                if "lm_head.weight" in dropped and lm_head_skeleton is not None:
                    embed_key = next((k for k in available if k.endswith("embed_tokens.weight")), None)
                    if embed_key is not None and embed_key not in hf_state:
                        hf_state[embed_key] = torch.empty_like(lm_head_skeleton)

        dcp.load(hf_state, storage_reader=sd_adapter.get_hf_storage_reader(path, False))
        tt_state = sd_adapter.from_hf(hf_state)

        buffer_names = [n for part in parts for n, _ in part.named_buffers()]
        unloaded: list[str] = []
        kept = 0
        for part in parts:
            own = set(part.state_dict().keys())
            result = part.load_state_dict(tt_state, strict=False)
            missing_own = [k for k in result.missing_keys if k in own]
            unloaded += unloaded_parameters(missing_own, buffer_names)
            kept += len(missing_own)
        if unloaded:
            raise RuntimeError(
                f"HF checkpoint {path} did not populate {len(unloaded)} key(s), e.g. {unloaded[:5]}"
            )
        logger.info(
            f"Loaded HF weights from {path}: {len(hf_state)} tensors requested, "
            f"{kept} runtime buffer(s) kept their initialized values"
        )
        self._sd_adapter = sd_adapter

    def hf_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """HF-named tensors, materialized one at a time, for the engine push.

        Under PP each rank yields only its own stages' tensors; every tensor
        exists on exactly one stage, so the union across ranks is the model.
        """
        from miles.backends.fsdp_utils.dtensor import gather_full_param

        for name, tensor in self._sd_adapter.to_hf(self._local_state_dict()).items():
            yield name, gather_full_param(tensor)

    # ------------------------------------------------------------------ infra

    def build_ref_runner(self, ref_load_path: str) -> LinearStepRunner:
        """A frozen second copy for reference log probs, CPU-offloaded so the
        two models never both hold HBM. Returned as a forward-only step runner
        so the shared loop drives it exactly like the actor."""
        if not ref_load_path:
            raise ValueError("--ref-load is required to build a torchtitan reference model")
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "reference model under pipeline parallelism needs a second schedule; unsupported yet"
            )
        parts, _, _, _ = self._build_parts(cpu_offload=True)
        self.load_hf(ref_load_path, parts=parts)
        for part in parts:
            part.eval()
            part.requires_grad_(False)
        logger.info(f"Built a CPU-offloaded torchtitan reference model from {ref_load_path}")
        return LinearStepRunner(lambda batch: self._forward(batch, module=parts[0]))

    def parallel_state(self):
        return create_titan_parallel_state(self.parallel_dims)

    def checkpoint_parts(self) -> dict:
        """torchtitan's optimizer and LR-scheduler containers are already Stateful."""
        return {
            "model": _TitanModelState(self.model_parts),
            "optimizer": self.optimizers,
            "lr_scheduler": self.lr_schedulers,
        }


class _TitanModelState(Stateful):
    """Stateful over the model parts for miles' DCP checkpoints.

    Plain ``state_dict()``/``load_state_dict()`` like everything else in this
    module: the generic ``get_state_dict`` helper walks module attributes and
    breaks on titan's fused-module state-dict hooks. Save and load run the same
    code, so every key is expected to land; the buffer allowance only forgives
    keys ``init_states`` can rebuild, anything else raises.
    """

    def __init__(self, model_parts: list):
        self.model_parts = model_parts

    def state_dict(self) -> dict:
        return {"model": {k: v for part in self.model_parts for k, v in part.state_dict().items()}}

    def load_state_dict(self, state_dict: dict) -> None:
        state = state_dict["model"]
        buffer_names = [n for part in self.model_parts for n, _ in part.named_buffers()]
        for part in self.model_parts:
            own = set(part.state_dict().keys())
            result = part.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
            unloaded = unloaded_parameters([k for k in result.missing_keys if k in own], buffer_names)
            if unloaded:
                raise RuntimeError(f"checkpoint left {len(unloaded)} key(s) unloaded, e.g. {unloaded[:5]}")

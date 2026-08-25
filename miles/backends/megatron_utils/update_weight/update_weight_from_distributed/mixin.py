import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.timer import timer

from ...lora_utils import build_lora_sync_config
from ..common import begin_weight_update, end_weight_update, weight_update_selector
from ..hf_weight_iterator import get_hf_weight_iterator

logger = logging.getLogger(__name__)


class DistBucketedWeightUpdateMixin:
    """Distributed weight-update lifecycle over the HF weight iterator.

    Consuming classes set args/model/model_name/quantization_config,
    weight_version, rollout_engines, _group_name, and — at connect time —
    is_sender / is_lora_sender. They implement
    ``_update_weight_implementation(bucket, pbar)`` plus the LoRA variants, and
    may override ``_after_base_weights`` (e.g. await in-flight writes).
    """

    def _init_weight_transfer(
        self,
        *,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        model_name: str,
        quantization_config: dict | None,
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        is_lora: bool,
    ) -> None:
        """Create the weight iterator and LoRA state. Call from subclass ``__init__``."""
        self.weights_getter = weights_getter
        # The distributed protocols only need TP/EP gathered; bridge still forces a full gather.
        self._hf_weight_iterator = get_hf_weight_iterator(
            args,
            model,
            required_placement=WeightUpdatePlacement(gather_pp=False),
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.is_lora = is_lora
        # Set by the actor before each update_weights call (loaded map at reconcile).
        self.multi_lora_adapters = None
        if self.is_lora:
            assert args.megatron_to_hf_mode == "bridge", (
                "LoRA weight sync over distributed engines requires "
                f"--megatron-to-hf-mode bridge (got {args.megatron_to_hf_mode!r})."
            )
            self._lora_config = build_lora_sync_config(args)
            self._lora_loaded = False

    def _update_lora_weights(self) -> None:
        """Orchestrate the LoRA adapter update; delegate transmit to the subclass.

        Mirrors the base path's split: this method owns the transport-agnostic
        steps (source gating and the unload-before-reload), and hands the
        gathered adapter to ``self._update_lora_weight_implementation`` —
        broadcast (NCCL) or p2p provide their own.

        All ranks call the iterator (required for internal TP collectives), but
        only the source rank transmits.
        """
        named_tensors = self._hf_weight_iterator.get_hf_lora_weights()

        if not self.is_lora_sender:
            return

        if self._lora_loaded:
            ray.get(
                [engine.unload_lora_adapter.remote(lora_name=LORA_ADAPTER_NAME) for engine in self.rollout_engines]
            )
        self._update_lora_weight_implementation(named_tensors)
        self._lora_loaded = True

    def _update_multi_lora_weights(self) -> None:
        """Upsert the actor-selected adapters; the push set is identical on every rank so TP collectives align."""
        adapters = self.multi_lora_adapters
        assert adapters is not None, "actor must set multi_lora_adapters before update_weights"
        for name in sorted(adapters):
            self._send_one_multi_lora_adapter(adapters[name])

    def _send_one_multi_lora_adapter(self, adapter) -> None:
        """All ranks call the iterator (TP collectives); only the source
        rank transmits."""
        from miles.utils.multi_lora import slot_lora_name

        lora_config = build_lora_sync_config(self.args) | {
            "r": adapter.config.rank,
            "lora_alpha": adapter.config.alpha,
        }

        named_tensors = self._hf_weight_iterator.get_hf_lora_weights(adapter)

        if not self.is_lora_sender:
            return

        self._update_multi_lora_weight_implementation(
            named_tensors,
            lora_name=slot_lora_name(adapter.slot),
            lora_config=lora_config,
        )

    def _after_base_weights(self) -> None:
        """Hook after the base-weight stream completes (e.g. await in-flight writes)."""

    def _pause_and_prepare_engines(self) -> None:
        """Pause rollout engines, flush cache, and open the weight-update session."""
        self._weight_update_selector = weight_update_selector(self.args)
        if dist.get_rank() == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            if mode != "in_place":
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            begin_weight_update(self.rollout_engines, self._weight_update_selector)

    def _finalize_and_resume_engines(self) -> None:
        """Close the weight-update session and resume rollout engines."""
        if dist.get_rank() == 0:
            # unify update weight version here to cover both full param and lora update
            ray.get(
                [
                    engine.update_weight_version.remote(weight_version=str(self.weight_version))
                    for engine in self.rollout_engines
                ]
            )
            end_weight_update(self.rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear ``update_weight_metrics``. Drained by the actor onto the step log;
        empty unless the updater recorded metrics during the last ``update_weights`` call."""
        out = self.__dict__.pop("update_weight_metrics", {})
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        """Orchestrate the full weight-update lifecycle.

        Full: pause → iterate HF buckets (senders transmit, other ranks join the
        gathers) → resume. LoRA: pause → adapter push → resume; the frozen base
        is never pushed (engines load it from ``hf_checkpoint`` at init).
        Progress is shown on sender ranks.
        """
        self.weight_version += 1

        self._pause_and_prepare_engines()
        dist.barrier(group=get_gloo_group())

        with timer("update_weights_implementation"):
            from miles.utils.multi_lora import is_multi_lora_enabled

            is_lora = getattr(self, "is_lora", False)
            is_multi_lora = is_lora and is_multi_lora_enabled(self.args)

            # LoRA: base weights are frozen and already loaded by the rollout engines
            # from ``hf_checkpoint``, so only full-param runs sync the base.
            if not is_lora:
                pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self.is_sender else None

                weights = self.weights_getter()
                for bucket in self._hf_weight_iterator.iter_hf_base_weights(weights, materialize=self.is_sender):
                    if self.is_sender:
                        self._update_weight_implementation(bucket, pbar)
                self._after_base_weights()
                dist.barrier(group=get_gloo_group())

            # Adapter weights: every iteration.
            if is_lora:
                if is_multi_lora:
                    self._update_multi_lora_weights()
                else:
                    self._update_lora_weights()
                dist.barrier(group=get_gloo_group())

        with timer("finalize_and_resume_engines"):
            self._finalize_and_resume_engines()
            dist.barrier(group=get_gloo_group())

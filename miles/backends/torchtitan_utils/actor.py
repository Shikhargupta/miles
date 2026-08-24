"""TrainRayActor over a torchtitan model.

Deliberately shaped like FSDPTrainRayActor: the RL flow (rollout data in ->
reference/actor log probs -> advantages -> optimizer steps -> weight sync) is the
shared code in ``training_utils``, and this class supplies only what is
torchtitan-specific -- model construction, its optimizer/LR containers, its
checkpointer, and its HF state-dict mapping.
"""

import logging
from argparse import Namespace

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.torchtitan_utils import compat
from miles.backends.torchtitan_utils.model import (
    build_engine_config,
    build_model,
    load_hf_weights,
    resolve_model_spec,
)
from miles.backends.torchtitan_utils.parallel import build_parallel_dims, create_titan_parallel_state
from miles.backends.torchtitan_utils.weight_bridge import TitanUpdateWeightFromTensor
from miles.backends.training_utils.ci_utils import check_grad_norm
from miles.backends.training_utils.data import get_batch, get_data_iterator, get_rollout_data
from miles.backends.training_utils.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_rollout_data,
    log_train_step,
)
from miles.backends.training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, loss_function
from miles.backends.training_utils.model_assets import load_model_assets
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.weight_sync import connect_engines_if_stale, verify_engine_weight_version
from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.ray_utils import Box
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)

_FORWARD_KEYS = [
    "tokens",
    "loss_masks",
    "total_lengths",
    "response_lengths",
    "max_seq_lens",
]
_TRAIN_KEYS = _FORWARD_KEYS + [
    "log_probs",
    "advantages",
    "returns",
    "ref_log_probs",
    "rollout_log_probs",
]


class TorchtitanTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info=None,
    ) -> int | None:  # type: ignore[override]
        compat.install()
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        assert recv_ckpt_src_rank is None, "torchtitan backend does not support checkpoint healing"
        assert not with_ref, "torchtitan backend does not support a reference model yet"
        assert not with_opd_teacher, "torchtitan backend does not support on-policy distillation yet"

        parallel_dims = build_parallel_dims(args)
        set_parallel_state(create_titan_parallel_state(parallel_dims))
        torch.manual_seed(args.seed)

        self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}
        if args.debug_rollout_only:
            return 0

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)
        self.prof = TrainProfiler(args)

        assets = load_model_assets(args)
        self.hf_config = assets.hf_config
        self.tokenizer = assets.tokenizer

        device = torch.device(f"cuda:{int(dist.get_rank() % args.num_gpus_per_node)}")
        spec = resolve_model_spec(args)
        config = build_engine_config(args, spec)
        model_config, self.model = build_model(args, spec, config, parallel_dims, device)
        self.sd_adapter = load_hf_weights(spec, model_config, self.model, args.hf_checkpoint)

        self.optimizers = config.optimizer.build(model_parts=[self.model])
        if spec.post_optimizer_build_fn is not None:
            spec.post_optimizer_build_fn(self.optimizers, [self.model], parallel_dims)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers, training_steps=max(args.num_rollout, 1)
        )
        self.titan_config = config

        self.weight_updater = TitanUpdateWeightFromTensor(args, self.model, self.sd_adapter)

        clear_memory()
        if args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return int(getattr(args, "start_rollout_id", None) or 0)

    @timer
    def sleep(self) -> None:
        if not self.args.offload_train:
            return
        print_memory("before offload model")
        self.model.cpu()
        _move_optimizer_state(self.optimizers, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return
        self.model.cuda()
        _move_optimizer_state(self.optimizers, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.args.save is None:
            return
        raise NotImplementedError("torchtitan checkpoint saving is not wired up yet")

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: Box,
        witness_info=None,
        attempt: int = 0,
    ) -> None:
        assert witness_info is None and attempt == 0
        self._heartbeat.bump()
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref, witness_info=None)
            with store_get_result:
                if self.args.debug_rollout_only:
                    return
                self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        self._heartbeat.bump()

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        data_iterators, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        data_iterator = data_iterators[0]
        assert num_microbatches, f"empty microbatch schedule for micro_batch_size={self.args.micro_batch_size}"

        rollout_data.update(self._compute_log_probs(data_iterator, num_microbatches))
        compute_advantages_and_returns(self.args, rollout_data)
        log_rollout_data(rollout_id, self.args, rollout_data)

        with timer("actor_train"):
            self._run_optimizer_steps(rollout_id, data_iterator, num_microbatches)

        self.prof.step(rollout_id=rollout_id)

    def _forward_logits(self, batch: dict) -> torch.Tensor:
        """One model forward. ``positions`` restart per document, which is how
        torchtitan's masked attention backends identify document boundaries; the
        sdpa backend ignores them beyond RoPE and applies a plain causal mask,
        which is why a microbatch may hold only one document (see
        validate_torchtitan_args)."""
        return self.model(batch["tokens"], positions=batch["position_ids"])

    @torch.no_grad()
    def _compute_log_probs(self, data_iterator, num_microbatches: list[int]) -> dict:
        forward_store = []
        data_iterator.reset()
        with timer("log_probs"):
            for step_id in range(len(num_microbatches)):
                for _ in self.prof.iterate_train_log_probs(
                    tqdm(range(num_microbatches[step_id]), desc="log_probs", disable=dist.get_rank() != 0)
                ):
                    batch = get_batch(
                        data_iterator,
                        _FORWARD_KEYS,
                        self.args.data_pad_size_multiplier,
                        self.args.qkv_format,
                        get_position_ids=True,
                    )
                    result = get_log_probs_and_entropy(
                        logits=self._forward_logits(batch),
                        args=self.args,
                        unconcat_tokens=batch["unconcat_tokens"],
                        total_lengths=batch["total_lengths"],
                        response_lengths=batch["response_lengths"],
                        with_entropy=True,
                        max_seq_lens=batch.get("max_seq_lens"),
                    )
                    entry = {"log_probs": result["log_probs"]}
                    if "entropy" in result:
                        entry["entropy"] = result["entropy"]
                    forward_store.append(entry)
        return aggregate_forward_results(forward_store, data_iterator, self.args, "")

    def _run_optimizer_steps(self, rollout_id: int, data_iterator, num_microbatches: list[int]) -> None:
        data_iterator.reset()
        num_steps = len(num_microbatches)
        for step_id in range(num_steps):
            self.optimizers.zero_grad(set_to_none=True)
            losses = []
            for _ in self.prof.iterate_train_actor(
                tqdm(range(num_microbatches[step_id]), desc="actor_train", disable=dist.get_rank() != 0)
            ):
                batch = get_batch(
                    data_iterator,
                    _TRAIN_KEYS,
                    self.args.data_pad_size_multiplier,
                    self.args.qkv_format,
                    get_position_ids=True,
                )
                loss, _normalizer, log_dict = loss_function(
                    args=self.args,
                    batch=batch,
                    num_microbatches=num_microbatches[step_id],
                    logits=self._forward_logits(batch),
                    apply_megatron_loss_scaling=False,
                )
                loss.backward()
                losses.append(log_dict)

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            grad_norm = grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else grad_norm.item()
            self.optimizers.step()
            self.lr_schedulers.step()

            if self.args.ci_test:
                check_grad_norm(
                    args=self.args,
                    grad_norm=grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    role="actor",
                    rank=get_parallel_state().intra_dp_cp.rank,
                )

            log_train_step(
                args=self.args,
                loss_dict=aggregate_train_losses(losses),
                grad_norm=grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                num_steps_per_rollout=num_steps,
                role="actor",
                extra_metrics=self.lr_schedulers.get_metrics(),
            )

    @timer
    def update_weights(self, info) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        connect_engines_if_stale(self.weight_updater, self.rollout_manager, info)
        self.weight_updater.update_weights()
        if dist.get_rank() == 0:
            import ray

            ray.get(self.rollout_manager.set_weight_version.remote(self.weight_updater.weight_version))
        if self.args.ci_test:
            verify_engine_weight_version(self.weight_updater, info.rollout_engines)
        clear_memory()

    def _get_parallel_config(self):
        return self.train_parallel_config


@torch.no_grad()
def _move_optimizer_state(optimizers, device: str) -> None:
    for optimizer in optimizers.optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)
    torch.cuda.synchronize()

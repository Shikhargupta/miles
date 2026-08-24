"""TrainRayActor over a torchtitan model.

Deliberately shaped like FSDPTrainRayActor: the RL flow (rollout data in ->
reference/actor log probs -> advantages -> optimizer steps -> weight sync) is
the shared code in ``training_utils``, and everything torchtitan lives behind
``TitanEngine``. This class never imports torchtitan: it hands the engine to
the shared loop as a step runner and wires miles' checkpointing, offload, and
weight-sync around it.
"""

import logging
from argparse import Namespace

import ray
import torch
import torch.distributed as dist

from miles.backends.torchtitan_utils.engine import TitanEngine
from miles.backends.torchtitan_utils.parallel import build_parallel_dims, create_titan_parallel_state
from miles.backends.torchtitan_utils.weight_bridge import TitanUpdateWeightFromTensor
from miles.backends.training_utils import checkpoint
from miles.backends.training_utils.data import get_data_iterator, get_rollout_data
from miles.backends.training_utils.log_utils import log_rollout_data
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.model_assets import load_model_assets
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native_loop import run_log_probs, run_optimizer_steps
from miles.backends.training_utils.weight_sync import connect_engines_if_stale, verify_engine_weight_version
from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, move_optimizer_state, print_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.ray_utils import Box
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


def _lr_total_steps(args: Namespace) -> int:
    """Optimizer steps over the whole run, for the LR schedule's horizon.

    Each rollout contributes ``rollout_batch_size * n_samples_per_prompt``
    samples consumed in optimizer steps of ``global_batch_size``. Using
    ``num_rollout`` alone would compress the schedule by that factor.
    """
    steps_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return args.num_rollout * max(steps_per_rollout, 1)


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
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        assert recv_ckpt_src_rank is None, "torchtitan backend does not support checkpoint healing"
        assert not with_opd_teacher, "torchtitan backend does not support on-policy distillation yet"

        parallel_dims = build_parallel_dims(args)
        set_parallel_state(create_titan_parallel_state(parallel_dims))
        torch.manual_seed(args.seed)

        self.train_parallel_config = {"dp_size": get_parallel_state().intra_dp.size}
        self.ref_runner = None
        if args.debug_rollout_only:
            return 0

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)
        self.prof = TrainProfiler(args)

        assets = load_model_assets(args)
        self.hf_config = assets.hf_config
        self.tokenizer = assets.tokenizer

        # TrainRayActor.init already selected this rank's device from LOCAL_RANK;
        # deriving it from the global rank would assume a contiguous rank->GPU
        # mapping, which Ray does not promise and multi-node breaks outright.
        device = torch.device(torch.cuda.current_device())
        self.engine = TitanEngine(
            args, device, lr_total_steps=_lr_total_steps(args), parallel_dims=parallel_dims
        )
        self.engine.load_hf(args.hf_checkpoint)

        # Built after the actor so the two never race for HBM during init; it is
        # CPU-offloaded, so it costs host memory rather than device memory.
        if with_ref:
            self.ref_runner = self.engine.build_ref_runner(args.ref_load)

        self.weight_updater = TitanUpdateWeightFromTensor(args, self.engine)

        self.global_step = 0
        self.micro_step = 0
        checkpoint.finalize_load(self, checkpoint.load(self))

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
        for part in self.engine.model_parts:
            part.cpu()
        move_optimizer_state(self.engine.optimizers.optimizers, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return
        for part in self.engine.model_parts:
            part.cuda()
        move_optimizer_state(self.engine.optimizers.optimizers, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def checkpoint_parts(self):
        return self.engine.checkpoint_parts()

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.args.save is None:
            return
        assert not self.args.async_save, "TorchtitanTrainRayActor does not support async_save yet."
        checkpoint.save(self, rollout_id)

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
        data_iterators, num_microbatches = get_data_iterator(self.args, self.engine.model_parts, rollout_data)
        data_iterator = data_iterators[0]
        assert num_microbatches, f"empty microbatch schedule for micro_batch_size={self.args.micro_batch_size}"

        if self.ref_runner is not None:
            rollout_data.update(
                run_log_probs(
                    self.args,
                    data_iterator,
                    num_microbatches,
                    self.ref_runner,
                    profiler=self.prof,
                    store_prefix="ref_",
                )
            )

        rollout_data.update(
            run_log_probs(
                self.args,
                data_iterator,
                num_microbatches,
                self.engine,
                profiler=self.prof,
            )
        )
        compute_advantages_and_returns(self.args, rollout_data)
        log_rollout_data(rollout_id, self.args, rollout_data)

        with timer("actor_train"):
            run_optimizer_steps(
                self.args,
                rollout_id,
                data_iterator,
                num_microbatches,
                self.engine,
                profiler=self.prof,
            )

        self.prof.step(rollout_id=rollout_id)

    @timer
    def update_weights(self, info) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        connect_engines_if_stale(self.weight_updater, self.rollout_manager, info)
        self.weight_updater.update_weights()
        if dist.get_rank() == 0:
            ray.get(self.rollout_manager.set_weight_version.remote(self.weight_updater.weight_version))
        if self.args.ci_test:
            verify_engine_weight_version(self.weight_updater, info.rollout_engines)
        clear_memory()

    def _get_parallel_config(self):
        return self.train_parallel_config

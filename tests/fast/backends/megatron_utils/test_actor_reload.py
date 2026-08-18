from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest


def _args(tmp_path: Path, **overrides) -> Namespace:
    defaults = dict(
        load=str(tmp_path / "pretrain"),
        save=str(tmp_path / "run"),
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
        rematerialize_param_from_master_weight=False,
        non_persistent_ckpt_type=None,
        debug_rollout_only=False,
        async_save=False,
        offload_train=False,
        keep_old_actor=False,
        use_pytorch_profiler=False,
        record_memory_history=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_checkpoint(directory: Path, *, iteration: int) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    return str(directory)


def _actor_module():
    return pytest.importorskip("miles.backends.megatron_utils.actor")


def _actor(*, role: str, args: Namespace):
    module = _actor_module()
    actor = module.MegatronTrainRayActor.__new__(module.MegatronTrainRayActor)
    actor.role = role
    actor.args = args
    actor._init_called = True
    actor._asleep = False
    actor._enable_weight_backup = False
    actor.with_ref = False
    actor.with_opd_teacher = False
    actor.model = None
    actor.optimizer = None
    actor.opt_param_scheduler = None
    return actor


def _watch_the_load(monkeypatch, *, args: Namespace, iteration: int) -> dict[str, Any]:
    model_module = pytest.importorskip("miles.backends.megatron_utils.model")
    seen: dict[str, Any] = {}

    def fake_load_checkpoint(*_args: Any, **_kwargs: Any) -> tuple[int, int]:
        seen["args_during_load"] = vars(args).copy()
        return iteration, 0

    monkeypatch.setattr(model_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(model_module, "clear_memory", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda *a, **k: None)
    monkeypatch.setattr(_actor_module(), "clear_memory", lambda *a, **k: None)
    return seen


class TestTheCheckpointAReloadRollsBackTo:
    def test_a_reload_reads_the_directory_this_trainer_saves_into(self, tmp_path, monkeypatch):
        """A take-over has to roll the trainer back to where the run really is, which is its own newest checkpoint."""
        save = _write_checkpoint(tmp_path / "run", iteration=50)
        args = _args(tmp_path)
        seen = _watch_the_load(monkeypatch, args=args, iteration=50)

        assert _actor(role="actor", args=args).load_state() == 51
        assert seen["args_during_load"]["load"] == save

    def test_an_offloaded_trainer_is_woken_before_anything_is_read_into_it(self, tmp_path, monkeypatch):
        """Nothing should be able to put a reloadable trainer to sleep, but waking it is cheaper than finding out."""
        _write_checkpoint(tmp_path / "run", iteration=50)
        args = _args(tmp_path)
        _watch_the_load(monkeypatch, args=args, iteration=50)
        actor = _actor(role="actor", args=args)
        actor._asleep = True
        woken: list[bool] = []
        monkeypatch.setattr(actor, "wake_up", lambda: woken.append(True))

        actor.load_state()

        assert woken == [True]

    def test_a_reload_forgets_the_rollout_the_previous_script_was_last_on(self, tmp_path, monkeypatch):
        """A freshly started trainer has not trained yet, and a reloaded one has to look the same to the next one."""
        _write_checkpoint(tmp_path / "run", iteration=50)
        args = _args(tmp_path)
        _watch_the_load(monkeypatch, args=args, iteration=50)
        actor = _actor(role="actor", args=args)
        actor._last_rollout_id = 49

        actor.load_state()

        assert not hasattr(actor, "_last_rollout_id")

    def test_a_reload_clears_the_flags_a_cold_started_parse_left_behind(self, tmp_path, monkeypatch):
        """A parse that found no checkpoint set these, and a reload onto a real checkpoint has to load all of it."""
        _write_checkpoint(tmp_path / "run", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=3)
        seen = _watch_the_load(monkeypatch, args=args, iteration=50)

        _actor(role="actor", args=args).load_state()

        during = seen["args_during_load"]
        assert (during["finetune"], during["no_load_optim"], during["no_load_rng"], during["ckpt_step"]) == (
            False,
            False,
            False,
            None,
        )

    def test_a_cold_started_parse_does_not_make_a_real_resume_start_over(self, tmp_path, monkeypatch):
        """The rollout to resume at is worked out under the overridden arguments, not the restored ones."""
        _write_checkpoint(tmp_path / "run", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True)
        _watch_the_load(monkeypatch, args=args, iteration=50)

        assert _actor(role="actor", args=args).load_state() == 51

    def test_a_reload_leaves_the_arguments_as_it_found_them(self, tmp_path, monkeypatch):
        """The override says where this one load reads from; the run's own arguments have to survive it."""
        _write_checkpoint(tmp_path / "run", iteration=50)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=3)
        _watch_the_load(monkeypatch, args=args, iteration=50)

        _actor(role="actor", args=args).load_state()

        assert args.load == str(tmp_path / "pretrain")
        assert (args.finetune, args.no_load_optim, args.no_load_rng, args.ckpt_step) == (True, True, True, 3)

    def test_a_critic_reload_reads_the_critic_directory(self, tmp_path, monkeypatch):
        """A critic's own arguments carry its checkpoint dirs, so reading --save off them reads the critic's."""
        critic_save = _write_checkpoint(tmp_path / "run_critic", iteration=60)
        args = _args(tmp_path, save=critic_save)
        seen = _watch_the_load(monkeypatch, args=args, iteration=60)

        assert _actor(role="critic", args=args).load_state() == 61
        assert seen["args_during_load"]["load"] == critic_save

    def test_a_reload_that_would_cold_start_is_refused(self, tmp_path):
        """The trainer is alive at the rollout the run reached, so pretrained weights here would replay it from 0."""
        with pytest.raises(AssertionError):
            _actor(role="actor", args=_args(tmp_path)).load_state()


class TestWhatAReloadRefuses:
    @pytest.mark.parametrize(
        "overrides",
        [
            dict(debug_rollout_only=True),
            dict(lora_rank=8),
            dict(multi_lora=True),
            dict(colocate=True),
            dict(rematerialize_param_from_master_weight=True),
            dict(non_persistent_ckpt_type="local"),
            dict(offload_train=True),
            dict(use_pytorch_profiler=True),
            dict(record_memory_history=True),
        ],
    )
    def test_a_trainer_it_cannot_restore_is_refused_before_anything_is_woken(self, tmp_path, overrides):
        """Each of these means a checkpoint load cannot put the trainer back where the run really is."""
        actor = _actor(role="actor", args=_args(tmp_path, **overrides))

        with pytest.raises(AssertionError):
            actor.load_state()

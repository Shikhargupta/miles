import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest


def _args(tmp_path: Path, **overrides) -> Namespace:
    defaults = dict(
        load=str(tmp_path / "pretrain"),
        save=str(tmp_path / "run"),
        ckpt_step=None,
        ref_ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        lora_rank=0,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
        rematerialize_param_from_master_weight=False,
        non_persistent_ckpt_type=None,
        fp16=False,
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
        offload_optimizer_states=False,
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


def _stub_the_reset(monkeypatch) -> None:
    monkeypatch.setattr(_actor_module(), "set_random_seed_from_args", lambda *a, **k: None)
    monkeypatch.setattr(_actor_module(), "reset_optimizer_state", lambda *a, **k: None)


class TestAReloadThatFindsNothingItSaved:
    def test_it_puts_the_trainer_back_where_the_run_began(self, tmp_path, monkeypatch, caplog):
        """A run that has not saved yet still has a state to go back to: the one it was started from."""
        _write_checkpoint(tmp_path / "pretrain", iteration=0)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True)
        seen = _watch_the_load(monkeypatch, args=args, iteration=0)
        actor = _actor(role="actor", args=args)
        _stub_the_reset(monkeypatch)

        with caplog.at_level(logging.INFO):
            assert actor.load_state() == 0

        assert seen["args_during_load"]["load"] == str(tmp_path / "pretrain")
        assert "found no checkpoint" in caplog.text

    def test_it_loads_under_the_arguments_the_run_was_started_with(self, tmp_path, monkeypatch):
        """Anything else would answer a different question than the one the trainer's own init answered."""
        _write_checkpoint(tmp_path / "pretrain", iteration=0)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=3)
        seen = _watch_the_load(monkeypatch, args=args, iteration=0)
        actor = _actor(role="actor", args=args)
        _stub_the_reset(monkeypatch)

        actor.load_state()

        during = seen["args_during_load"]
        assert (during["finetune"], during["no_load_optim"], during["no_load_rng"], during["ckpt_step"]) == (
            True,
            True,
            True,
            3,
        )

    def test_it_reseeds_the_rng_and_resets_the_optimizer(self, tmp_path, monkeypatch):
        """A checkpoint load overwrites neither, so a live trainer would keep both from the rollouts it discards."""
        _write_checkpoint(tmp_path / "pretrain", iteration=0)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True)
        _watch_the_load(monkeypatch, args=args, iteration=0)
        actor = _actor(role="actor", args=args)
        actor.optimizer = "the live optimizer"
        reseeded: list[Namespace] = []
        reset: list[str] = []
        monkeypatch.setattr(_actor_module(), "set_random_seed_from_args", reseeded.append)
        monkeypatch.setattr(_actor_module(), "reset_optimizer_state", reset.append)

        actor.load_state()

        assert reseeded == [args] and reset == ["the live optimizer"]

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(finetune=False),
            dict(no_load_optim=False),
            dict(no_load_rng=False),
            dict(ckpt_step=7),
        ],
    )
    def test_a_run_that_did_not_cold_start_is_refused(self, tmp_path, monkeypatch, overrides):
        """Only a cold-started run is supported here; resetting state a load would also restore is undesigned."""
        _write_checkpoint(tmp_path / "pretrain", iteration=0)
        args = _args(tmp_path, finetune=True, no_load_optim=True, no_load_rng=True, **overrides)
        _watch_the_load(monkeypatch, args=args, iteration=0)
        _stub_the_reset(monkeypatch)

        with pytest.raises(AssertionError):
            _actor(role="actor", args=args).load_state()

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(fp16=True),
            dict(use_precision_aware_optimizer=True),
            dict(optimizer_cpu_offload=True),
            dict(offload_optimizer_states=True),
        ],
    )
    def test_a_trainer_whose_state_cannot_be_reset_is_refused(self, tmp_path, overrides):
        """These keep state a reset would have to rebuild, and a half-reset trainer is worse than a refused one."""
        actor = _actor(role="actor", args=_args(tmp_path, **overrides))

        with pytest.raises(AssertionError):
            actor.load_state()

    def test_a_reload_that_did_save_is_not_pushed_back_to_the_beginning(self, tmp_path, monkeypatch):
        """The reset path is for a run with nothing of its own, and running it over a real resume would undo it."""
        _write_checkpoint(tmp_path / "run", iteration=50)
        args = _args(tmp_path, fp16=True)
        _watch_the_load(monkeypatch, args=args, iteration=50)

        assert _actor(role="actor", args=args).load_state() == 51


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

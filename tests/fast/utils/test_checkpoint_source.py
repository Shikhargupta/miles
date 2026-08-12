from argparse import Namespace
from pathlib import Path

from miles.utils.arguments import (
    CHECKPOINT_SOURCE_DEFAULTS,
    capture_requested_checkpoint_source,
    resolve_checkpoint_source,
)
from miles.utils.megatron_config import _rebase_checkpoint_source_on_the_policy_dir, compute_policy_checkpoint_dir

_MODEL_ID = "policy_a"


def _checkpoint_dir(tmp_path: Path, name: str) -> str:
    path = str(tmp_path / name)
    _write_tracker(path)
    return path


def _write_tracker(load_dir: str) -> None:
    Path(load_dir).mkdir(parents=True, exist_ok=True)
    (Path(load_dir) / "latest_checkpointed_iteration.txt").write_text("41")


def _cold_start(tmp_path: Path, *, model_id: str | None = None, **overrides) -> Namespace:
    args = Namespace(
        **{
            "load": str(tmp_path / "ckpt"),
            "ckpt_step": None,
            "finetune": False,
            "no_load_optim": False,
            "no_load_rng": False,
            "critic_load": None,
            "ref_load": _checkpoint_dir(tmp_path, "ref"),
            "ref_ckpt_step": None,
            "hf_checkpoint": None,
            "megatron_to_hf_mode": "raw",
            "start_rollout_id": 0,
            **overrides,
        }
    )
    capture_requested_checkpoint_source(args)
    resolve_checkpoint_source(args)
    if model_id is not None:
        _rebase_checkpoint_source_on_the_policy_dir(args, model_id=model_id)
    return args


def _hot_restart(args: Namespace) -> Namespace:
    resolve_checkpoint_source(args)
    return args


def _source_of(args: Namespace) -> dict:
    return {name: getattr(args, name) for name in (*CHECKPOINT_SOURCE_DEFAULTS, "start_rollout_id")}


class TestAHotRestartResolvesWhatAColdStartWouldResolve:
    def test_a_single_policy_run_that_has_written_a_checkpoint_reloads_it(self, tmp_path):
        """The whole point of a reload is to start from the run's own checkpoint rather than the reference."""
        started_fresh = _cold_start(tmp_path)
        _write_tracker(str(tmp_path / "ckpt"))

        assert _source_of(_hot_restart(started_fresh)) == _source_of(_cold_start(tmp_path))
        assert started_fresh.load == str(tmp_path / "ckpt")

    def test_a_run_that_has_written_nothing_yet_still_falls_back_to_the_reference(self, tmp_path):
        """A hot restart before the first save has to land exactly where relaunching the command would."""
        started_fresh = _cold_start(tmp_path)

        assert _source_of(_hot_restart(started_fresh)) == _source_of(_cold_start(tmp_path))
        assert started_fresh.load == started_fresh.ref_load

    def test_a_policy_of_a_multi_policy_run_keeps_its_own_subdirectory(self, tmp_path):
        """Reloading the root would load whatever the run happened to leave there, not this policy's weights."""
        policy_dir = compute_policy_checkpoint_dir(str(tmp_path / "ckpt"), _MODEL_ID)
        started_fresh = _cold_start(tmp_path, model_id=_MODEL_ID)
        _write_tracker(policy_dir)

        assert _source_of(_hot_restart(started_fresh)) == _source_of(_cold_start(tmp_path, model_id=_MODEL_ID))
        assert started_fresh.load == policy_dir

    def test_a_policy_that_has_written_nothing_yet_falls_back_the_same_way_both_times(self, tmp_path):
        """A per-policy directory that does not exist yet must not resolve differently on the two paths."""
        started_fresh = _cold_start(tmp_path, model_id=_MODEL_ID)

        assert _source_of(_hot_restart(started_fresh)) == _source_of(_cold_start(tmp_path, model_id=_MODEL_ID))

    def test_a_critic_that_was_given_no_load_directory_re_derives_it_with_the_actor(self, tmp_path):
        """The critic's rollout id drives the whole loop, so a frozen fallback rolls the run back in silence."""
        started_fresh = _cold_start(tmp_path)
        assert started_fresh.critic_load == started_fresh.ref_load
        _write_tracker(str(tmp_path / "ckpt"))

        reloaded = _hot_restart(started_fresh)

        assert _source_of(reloaded) == _source_of(_cold_start(tmp_path))
        assert reloaded.critic_load == str(tmp_path / "ckpt")

    def test_a_critic_load_the_user_named_is_never_re_derived_away(self, tmp_path):
        """It is a requested value like `--load`, so a restart restores it rather than deriving over it."""
        named = _checkpoint_dir(tmp_path, "critic")
        started_fresh = _cold_start(tmp_path, critic_load=named)
        _write_tracker(str(tmp_path / "ckpt"))

        reloaded = _hot_restart(started_fresh)

        assert reloaded.critic_load == named
        assert _source_of(reloaded) == _source_of(_cold_start(tmp_path, critic_load=named))

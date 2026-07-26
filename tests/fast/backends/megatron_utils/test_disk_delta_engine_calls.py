from argparse import Namespace
from unittest.mock import MagicMock, patch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta import UpdateWeightFromDiskDelta

_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.delta"


class _RecordingApiClient:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def __getattr__(self, name: str):
        async def method(**kwargs):
            self._calls.append((name, kwargs))
            return {"success": True}

        return method


def _make_updater(calls: list[tuple[str, dict]]) -> UpdateWeightFromDiskDelta:
    updater = UpdateWeightFromDiskDelta.__new__(UpdateWeightFromDiskDelta)
    updater.args = Namespace(
        update_weight_local_checkpoint_dir="/local/ckpt",
        update_weight_disk_dir="/shared/delta",
        pause_generation_mode="retract",
        check_weight_update_equal=False,
    )
    updater.rollout_engines = [_RecordingApiClient(calls)]
    updater.weight_version = 7
    updater._post_write_hook = None
    updater._version_dir = "/shared/delta/v7"
    return updater


def test_reload_engines_pulls_with_both_checkpoint_dirs_then_reloads():
    """The client takes no args, so the updater must pass both dirs itself; missing either one
    used to be injected by the engine wrapper and would now be a TypeError at runtime."""
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater._reload_engines()

    assert [name for name, _kwargs in calls] == [
        "pull_weights",
        "pause_generation",
        "flush_cache",
        "update_weights_from_disk",
        "continue_generation",
    ]
    assert calls[1][1] == {"mode": "retract"}
    assert calls[0][1] == {
        "target_version": 7,
        "local_checkpoint_dir": "/local/ckpt",
        "source_dir": "/shared/delta",
    }
    assert calls[3][1] == {"model_path": "/local/ckpt", "load_format": None, "weight_version": "7"}


def test_in_place_pause_mode_skips_the_flush():
    """in_place pause keeps the running batches, so draining the queue would be wrong."""
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)
    updater.args.pause_generation_mode = "in_place"

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater._reload_engines()

    assert "flush_cache" not in [name for name, _kwargs in calls]


def test_non_source_rank_issues_no_requests():
    calls: list[tuple[str, dict]] = []
    updater = _make_updater(calls)

    with patch(f"{_MODULE}.dist") as dist_mock, patch(f"{_MODULE}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 1
        updater._reload_engines()

    assert calls == []

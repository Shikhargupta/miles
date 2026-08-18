from pathlib import Path
from types import SimpleNamespace

from miles.rollout.data_source import RolloutDataSource


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    assert source.load(rollout_id=3) is False
    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _bare_source(**overrides) -> RolloutDataSource:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = _make_args(**overrides)
    source.metadata = {}
    return source


def test_load_answers_false_when_it_finds_no_state(tmp_path: Path) -> None:
    """The caller that demands a state to resume from has to be able to tell that there was none."""
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    assert source.load(rollout_id=3) is False


def test_load_answers_true_when_it_reads_a_state(tmp_path: Path) -> None:
    """A take-over accepts the load only on this answer, so the successful path has to give it."""
    import torch

    from miles.rollout.data_source import compute_global_dataset_state_path

    path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_offset": 7, "epoch_id": 1}, path)
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    assert source.load(rollout_id=3) is True
    assert (source.sample_offset, source.epoch_id) == (7, 1)

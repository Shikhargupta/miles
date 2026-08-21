import contextlib
import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.ft.conftest_ft import app as app_module
from tests.e2e.ft.conftest_ft.app import _DUMPS_ROOT_ENV, RunSideRequest, resolve_dump_dir, run_pipeline
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils import command_utils


def test_dump_dir_hangs_off_the_configured_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cluster says where dumps go through the environment its infra file sets."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")

    assert resolve_dump_dir("scenario_x") == str(tmp_path / "dumps" / "run-a" / "scenario_x")


def test_an_empty_configured_root_is_not_a_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset variable and one set to nothing both mean the cluster configured no root."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, "")
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")
    monkeypatch.setattr("os.makedirs", lambda path, exist_ok: None)

    assert resolve_dump_dir("scenario_x") == "/node_public/dumps/run-a/scenario_x"


def test_two_runs_of_one_test_do_not_share_a_dump_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run id in the path is what stops one run's rmtree deleting another's dumps."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")
    first = resolve_dump_dir("scenario_x")
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-b")

    assert resolve_dump_dir("scenario_x") != first


def test_the_dump_directory_exists_when_it_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers write into the returned path without creating it themselves."""
    monkeypatch.setenv(_DUMPS_ROOT_ENV, str(tmp_path / "dumps"))
    monkeypatch.setenv("MILES_SCRIPT_RUN_ID", "run-a")

    assert Path(resolve_dump_dir("scenario_x")).is_dir()


def test_each_comparison_side_can_transform_its_config_before_the_context_and_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A side-specific release has to be shared by its target context and the launch it drives."""
    requests: list[RunSideRequest] = []
    contexts: list[command_utils.ExecuteTrainConfig] = []
    mode = FTTestMode(
        model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
    )

    @contextlib.contextmanager
    def target_context(_mode: FTTestMode, _dump_dir: str, config: command_utils.ExecuteTrainConfig) -> Iterator[None]:
        contexts.append(config)
        yield

    monkeypatch.setattr(app_module, "resolve_dump_dir", lambda _test_name: str(tmp_path / "comparison"))
    monkeypatch.setattr(app_module, "prepare", lambda _mode: None)
    monkeypatch.setattr(
        command_utils, "default_config", lambda: command_utils.ExecuteTrainConfig(run_id="shared-release")
    )

    run_pipeline(
        test_name="scenario_x",
        build_baseline_args=lambda *_args: "",
        build_target_args=lambda *_args: "",
        compare_fn=lambda *_args: None,
        phases=None,
        mode=None,
        target_side_context=target_context,
        config_for_side=lambda side, config: dataclasses.replace(config, run_id=f"{config.run_id}-{side}"),
        run_side=requests.append,
        resolve_mode_fn=lambda _mode: mode,
    )

    assert [request.config.run_id for request in requests] == ["shared-release-baseline", "shared-release-target"]
    assert len(contexts) == 1
    assert contexts[0] is requests[1].config

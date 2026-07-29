import os
import threading
from pathlib import Path

import pytest

from miles.utils.workers.command_actor import CommandActor


class _FakeExit:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.event = threading.Event()
        self.codes: list[int] = []
        monkeypatch.setattr(os, "_exit", self._exit)

    def _exit(self, code: int) -> None:
        self.codes.append(code)
        self.event.set()

    def wait(self) -> None:
        assert self.event.wait(timeout=10)


class TestRun:
    def test_launches_subprocess_with_envs_merged_over_parent_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The command sees the given envs, inherited parent vars, and envs override same-named parent vars."""
        fake_exit = _FakeExit(monkeypatch)
        monkeypatch.setenv("COMMAND_ACTOR_PARENT_VAR", "parent")
        monkeypatch.setenv("COMMAND_ACTOR_OVERRIDDEN_VAR", "from-parent")
        output_path = tmp_path / "output.txt"

        CommandActor().run(
            cmd=(
                'printf "%s %s %s" "$COMMAND_ACTOR_TEST_VAR" "$COMMAND_ACTOR_PARENT_VAR" '
                f'"$COMMAND_ACTOR_OVERRIDDEN_VAR" > {output_path}'
            ),
            envs={"COMMAND_ACTOR_TEST_VAR": "hello", "COMMAND_ACTOR_OVERRIDDEN_VAR": "from-envs"},
        )
        fake_exit.wait()

        assert output_path.read_text() == "hello parent from-envs"

    def test_rejects_second_run(self, monkeypatch: pytest.MonkeyPatch):
        """Calling run a second time on the same actor is rejected."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="true", envs={})

        with pytest.raises(AssertionError):
            actor.run(cmd="true", envs={})
        fake_exit.wait()


class TestLifecycleBinding:
    def test_exits_actor_process_with_zero_on_subprocess_success(self, monkeypatch: pytest.MonkeyPatch):
        """The actor process exits with code 0 when the subprocess succeeds."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd="true", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [0]

    def test_exits_actor_process_with_subprocess_returncode_on_failure(self, monkeypatch: pytest.MonkeyPatch):
        """The actor process exits with the subprocess returncode when it fails."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd="exit 7", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [7]

    def test_exits_actor_process_with_one_on_signal_killed_subprocess(self, monkeypatch: pytest.MonkeyPatch):
        """A signal-killed subprocess (negative returncode) maps to exit code 1."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd='kill -TERM "$$"', envs={})
        fake_exit.wait()

        assert fake_exit.codes == [1]

    def test_does_not_exit_while_subprocess_is_running(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """The actor keeps running until the subprocess actually exits."""
        fake_exit = _FakeExit(monkeypatch)
        flag_path = tmp_path / "flag"

        CommandActor().run(cmd=f'while [ ! -f "{flag_path}" ]; do sleep 0.01; done', envs={})

        assert not fake_exit.event.wait(timeout=0.3)
        flag_path.touch()
        fake_exit.wait()
        assert fake_exit.codes == [0]

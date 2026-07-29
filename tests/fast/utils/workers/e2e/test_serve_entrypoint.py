import pytest

from tests.fast.utils.workers.e2e.harness import READY_TIMEOUT_SECONDS, port_is_refused, reserve_port


class TestExecChain:
    async def test_the_served_process_is_the_spawned_one(self, handle, server):
        """execve keeps the pid, so terminating the spawned process really stops the server."""
        assert await handle.report_pid() == server.process.pid

    async def test_worker_argv_reaches_the_factory(self, handle):
        """Everything after -- is handed to the worker factory."""
        argv = await handle.report_argv()
        assert "--state-dir" in argv

    async def test_worker_argv_keeps_its_own_separator(self, spawn, make_handle):
        """Only the first -- splits, so worker argv may contain further separators."""
        server = spawn(worker_argv=["--flag", "--", "--inner"])
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        argv = await handle.report_argv()
        assert argv[-3:] == ["--flag", "--", "--inner"]

    async def test_env_var_hook_receives_worker_argv(self, handle):
        """The env-var hook is called with the worker argv, not the entrypoint argv."""
        recorded = await handle.report_env(name="MILES_E2E_ARGV")
        assert "--state-dir" in recorded

    async def test_env_var_hook_runs_before_the_heavy_imports(self, handle):
        """The bootstrap computes env vars before importing the server stack."""
        assert await handle.report_env(name="MILES_E2E_PREMATURE_IMPORTS") == ""

    async def test_parent_environment_is_inherited(self, spawn, make_handle):
        """Environment from the launcher reaches the worker."""
        server = spawn(extra_env={"MILES_E2E_MARKER": "inherited"})
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        assert await handle.report_env(name="MILES_E2E_MARKER") == "inherited"

    async def test_env_var_hook_is_optional(self, spawn, make_handle):
        """Serving without the hook still works."""
        server = spawn(env_var_fn=False)
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        assert await handle.demo_sync(a=1, b=1) == 2
        assert await handle.report_env(name="MILES_E2E_ARGV") is None


class TestStartupFailures:
    async def test_unknown_worker_path_fails_fast(self, spawn):
        """A worker path that cannot be imported exits instead of serving."""
        server = spawn(worker_path="no.such.module.make_worker", wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

    async def test_missing_worker_argument_is_a_usage_error(self, spawn):
        """argparse rejects a missing --worker with its usage exit code."""
        import os
        import subprocess
        import sys

        from tests.fast.utils.workers.e2e.harness import REPO_ROOT

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, "-m", "miles.utils.workers.serving.serve", "--host", "127.0.0.1"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            timeout=60,
        )

        assert result.returncode == 2
        assert b"usage" in result.stderr.lower()

    async def test_port_conflict_fails_fast(self, spawn, server):
        """A second server on a taken port exits without disturbing the first."""
        conflicting = spawn(port=server.port, wait=False)
        assert conflicting.wait(timeout=30.0) not in (None, 0)
        assert server.is_running()

    @pytest.mark.parametrize("bad_path", ["no_colon_module", "miles.utils.workers.serving.serve.no_such_attr"])
    async def test_bad_factory_paths_fail_fast(self, spawn, bad_path):
        """Malformed or missing factory paths exit rather than serving a broken worker."""
        server = spawn(worker_path=bad_path, wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)


class TestPortBinding:
    async def test_server_binds_only_the_requested_port(self, server):
        """The server listens where it was told and nowhere else."""
        assert port_is_refused(reserve_port())
        assert not port_is_refused(server.port)

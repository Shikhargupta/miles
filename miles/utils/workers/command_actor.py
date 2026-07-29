import logging
import os
import subprocess
import threading

from miles.utils.workers.node_probe import NodeProbeMixin

logger = logging.getLogger(__name__)


class CommandActor(NodeProbeMixin):
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    def run(self, cmd: str, envs: dict[str, str]) -> None:
        assert self._process is None, "CommandActor.run can only be called once"

        logger.info(f"CommandActor launches subprocess cmd={cmd!r} env_names={sorted(envs)}")
        self._process = subprocess.Popen(cmd, shell=True, env={**os.environ, **envs})

        threading.Thread(target=_babysit, args=(self._process,), daemon=True).start()


def _babysit(process: subprocess.Popen) -> None:
    returncode = process.wait()

    logger.info(f"CommandActor exits since its subprocess exited with returncode={returncode}")
    os._exit(returncode if 0 <= returncode <= 255 else 1)

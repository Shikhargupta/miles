import logging
import os
import subprocess

import ray

from miles.utils.misc import get_current_node_ip, get_free_port

logger = logging.getLogger(__name__)


class CommandActor:
    @staticmethod
    def _get_node_ip() -> str:
        return get_current_node_ip()

    @staticmethod
    def _get_free_consecutive_ports(*, start_port: int, consecutive: int) -> int:
        return get_free_port(start_port=start_port, consecutive=consecutive)

    def run(self, *, cmd: str, envs: dict[str, str]) -> None:
        logger.info(f"CommandActor starting subprocess: {cmd=} {envs=}")
        process = subprocess.Popen(cmd, shell=True, env={**os.environ, **envs})

        exit_code = process.wait()

        logger.info(f"CommandActor subprocess exited, exiting actor to keep lifetimes bound: {exit_code=} {cmd=}")
        ray.actor.exit_actor()

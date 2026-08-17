# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging

from tests.e2e.ft.conftest_ft.pod_deletion import KUBECTL_TIMEOUT_SECONDS

from miles.utils.external_utils.command_utils.common import run_process

logger = logging.getLogger(__name__)


def sigkill_process_patterns_in_pod(*, namespace: str, pod_name: str, container: str, process_pattern: str) -> None:
    result = run_process(
        [
            "kubectl",
            "exec",
            "--namespace",
            namespace,
            pod_name,
            "--container",
            container,
            "--",
            "pkill",
            "-9",
            "-f",
            process_pattern,
        ],
        capture_output=True,
        check=False,
        timeout=KUBECTL_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"No process matching {process_pattern!r} was killed inside {pod_name} (exit "
        f"{result.returncode}): {result.stderr.strip() or result.stdout.strip()}. A crash nobody caused would "
        f"otherwise be counted as one that happened"
    )

    logger.info(f"Sigkilled a {process_pattern} process inside pod {pod_name}")

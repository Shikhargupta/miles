from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.progress import TRAIN_STEP_METRIC_KEY

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import MetricEvent

DISCARDED_EVENTS_GLOB: str = ".trash_*"
CHECKPOINT_SNAPSHOT_GLOB: str = "iter_*/debug_events"


def read_step_events(events_dir: Path) -> dict[int, list[str]]:
    events_of_rollout_id: dict[int, list[str]] = {}
    for event in read_events(events_dir):
        if isinstance(event, MetricEvent) and event.rollout_id is not None and TRAIN_STEP_METRIC_KEY in event.metrics:
            events_of_rollout_id.setdefault(event.rollout_id, []).append(event.model_dump_json())
    return dict(sorted(events_of_rollout_id.items()))


def read_discarded_event_dirs(dump_dir: str) -> list[Path]:
    return sorted(Path(dump_dir).glob(DISCARDED_EVENTS_GLOB))


def read_checkpoint_snapshot_dirs(checkpoint_dir: str) -> list[Path]:
    return sorted(Path(checkpoint_dir).glob(CHECKPOINT_SNAPSHOT_GLOB))

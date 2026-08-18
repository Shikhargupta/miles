import logging
from dataclasses import dataclass
from pathlib import Path

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import MetricEvent

logger = logging.getLogger(__name__)

CHECKPOINT_TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"
TRAIN_STEP_METRIC_KEY: str = "train/grad_norm"


@dataclass(frozen=True)
class RunProgress:
    last_saved_iteration: int | None
    last_finished_rollout_id: int | None


def read_run_progress(*, checkpoint_dir: Path, events_dir: Path) -> RunProgress:
    return RunProgress(
        last_saved_iteration=read_last_saved_iteration(checkpoint_dir),
        last_finished_rollout_id=read_last_finished_rollout_id(events_dir),
    )


def read_last_saved_iteration(checkpoint_dir: Path) -> int | None:
    if not checkpoint_dir.is_dir():
        return None

    trackers = sorted(checkpoint_dir.glob(f"**/{CHECKPOINT_TRACKER_FILENAME}"))
    assert len(trackers) <= 1, (
        f"{checkpoint_dir} holds the trackers {[str(one) for one in trackers]}, so this run trains several policies, "
        f"and one restart window per run cannot be read off several trainers that save at their own pace"
    )
    if not trackers:
        return None

    text = _read_text_or_empty(trackers[0])
    return int(text) if text.isdigit() else None


def read_last_finished_rollout_id(events_dir: Path) -> int | None:
    return max(read_finished_rollout_ids(events_dir), default=None)


def read_finished_rollout_ids(events_dir: Path) -> list[int]:
    if not events_dir.is_dir():
        return []

    return sorted(
        {
            event.rollout_id
            for event in read_events(events_dir)
            if isinstance(event, MetricEvent)
            and event.rollout_id is not None
            and TRAIN_STEP_METRIC_KEY in event.metrics
        }
    )


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        logger.info(f"Failed to read {path}; treating it as not written yet", exc_info=True)
        return ""

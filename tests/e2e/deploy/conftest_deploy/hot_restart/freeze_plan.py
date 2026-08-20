import logging
import shlex
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_freeze_plan
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.test_utils.ft_test_actions import CI_FT_TEST_ACTIONS_PATH_FLAG, write_ft_test_actions

logger = logging.getLogger(__name__)

FREEZE_PLAN_DIRNAME: str = "hot_restart"


# TODO ad hoc hack: revert after the args refactor
def compute_freeze_plan_path(side_dump_dir: str) -> Path:
    side = Path(side_dump_dir)

    assert side.name in (BASELINE_SIDE, TARGET_SIDE), (
        f"the freeze plan is kept beside the two sides of one comparison, and {side_dump_dir} names neither of "
        f"them; a plan kept under a side's own dump directory is deleted when that side's run clears it"
    )
    return side.parent / FREEZE_PLAN_DIRNAME / f"{side.name}_freeze_plan.json"


# TODO ad hoc hack: revert after the args refactor
def with_the_freeze_plan_of(train_args: str, *, plan_path: Path) -> str:
    return f"{train_args}{CI_FT_TEST_ACTIONS_PATH_FLAG} {shlex.quote(str(plan_path))} "


# TODO ad hoc hack: revert after the args refactor
def write_freeze_plan(plan_path: Path, *, frozen_rollout_id: int | None) -> None:
    write_ft_test_actions(plan_path, compute_freeze_plan(frozen_rollout_id))
    logger.info(f"{plan_path} now freezes the run after step {frozen_rollout_id}")

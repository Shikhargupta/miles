import logging
import shlex
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_freeze_plan

from miles.utils.test_utils.ft_test_actions import CI_FT_TEST_ACTIONS_PATH_FLAG, write_ft_test_actions

logger = logging.getLogger(__name__)

FREEZE_PLAN_PATH: str = "hot_restart/freeze_plan.json"


# TODO ad hoc hack: revert after the args refactor
def compute_freeze_plan_path(dump_dir: str) -> Path:
    return Path(dump_dir) / FREEZE_PLAN_PATH


# TODO ad hoc hack: revert after the args refactor
def with_the_freeze_plan_of(train_args: str, *, plan_path: Path) -> str:
    return f"{train_args}{CI_FT_TEST_ACTIONS_PATH_FLAG} {shlex.quote(str(plan_path))} "


# TODO ad hoc hack: revert after the args refactor
def write_freeze_plan(plan_path: Path, *, frozen_rollout_id: int | None) -> None:
    write_ft_test_actions(plan_path, compute_freeze_plan(frozen_rollout_id))
    logger.info(f"{plan_path} now freezes the run after step {frozen_rollout_id}")

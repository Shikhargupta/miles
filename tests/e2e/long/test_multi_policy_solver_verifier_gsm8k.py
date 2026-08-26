import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import SOLVER_MODEL_ID, VERIFIER_MODEL_ID, ScriptArgs, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_multi_policy import TrainRewardBounds, execute

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=15000,
    suite="stage-c-4-gpu-h200",
    labels=["long"],
    disabled=(
        "the recipe was realigned with the single-policy GSM8K baseline (nonzero-std dynamic filter, 32x8 "
        "groups, response length 1024) and its acceptance has not been recalibrated against a run of that "
        "recipe yet. The historical stall around solver=99 verifier=80 is fixed: since the rollout-disposal "
        "and executor-teardown fixes, 100-rollout runs complete and the Ray job succeeds. What remains open "
        "is the learning gate: with the dynamic filter on, rollout/raw_reward is computed over accepted "
        "nonzero-std groups only, whose mean is pinned near .5 by construction, so growth on that metric no "
        "longer measures learning -- the per-policy held-out eval curves (eval/gsm8k/solver) do. Re-enable "
        "by running this recipe once, then gating on the observed eval trajectory."
    ),
)

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "250"))

# TODO: tighten these weak bounds once the e2e run has been observed.
TRAIN_REWARD_BOUNDS = {
    SOLVER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.3, min_growth=None),
    VERIFIER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.1, min_growth=None),
}


if __name__ == "__main__":
    args = dataclasses.replace(command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        args,
        wandb_args=command_utils.get_default_wandb_args(__file__),
        train_reward_bounds=TRAIN_REWARD_BOUNDS,
    )

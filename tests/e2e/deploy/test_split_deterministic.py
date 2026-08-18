from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.scenario_split_deterministic import run_ci

register_cuda_ci(
    est_time=2600,
    suite="stage-c-8-gpu-h200",
    labels=["deploy"],
)

if __name__ == "__main__":
    run_ci()

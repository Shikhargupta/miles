"""Contract tests for the shared CI policy and GitHub trigger adapter."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.ci.ci_policy import (
    NIGHTLY_CADENCE,
    REGULAR_CADENCE,
    WEEKLY_CADENCE,
    is_docs_only_change,
    resolve_policy,
    resolve_workflow_inputs,
)
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.mark.parametrize(
    ("event_name", "schedule", "labels_json", "cadence", "raw_labels", "bypass_fastfail"),
    [
        ("pull_request", "", "[]", REGULAR_CADENCE, (), False),
        (
            "pull_request",
            "",
            json.dumps(["run-ci-megatron", "bypass-fastfail"]),
            REGULAR_CADENCE,
            ("run-ci-megatron", "bypass-fastfail"),
            True,
        ),
        (
            "pull_request",
            "",
            json.dumps(
                [
                    "ignored",
                    "run-ci-megatron",
                    "nightly",
                    "run-ci-megatron",
                    "run-ci-a_B.c-d",
                    "run-ci-/unsafe",
                ]
            ),
            NIGHTLY_CADENCE,
            ("run-ci-megatron", "nightly", "run-ci-megatron", "run-ci-a_B.c-d"),
            True,
        ),
        ("schedule", "0 15 * * 0-5", "not JSON", NIGHTLY_CADENCE, (), True),
        ("schedule", "0 15 * * 6", "not JSON", WEEKLY_CADENCE, (), True),
        ("workflow_dispatch", "", "not JSON", REGULAR_CADENCE, (), False),
    ],
)
def test_trigger_facts_resolve_to_stable_workflow_outputs(
    event_name,
    schedule,
    labels_json,
    cadence,
    raw_labels,
    bypass_fastfail,
):
    policy = resolve_workflow_inputs(event_name, schedule, labels_json)

    assert policy.cadence == cadence
    assert policy.raw_labels == raw_labels
    assert policy.bypass_fastfail is bypass_fastfail


@pytest.mark.parametrize(
    ("event_name", "raw_labels", "changed_files", "docs_only"),
    [
        # The only combination that skips: PR, no labels, non-empty all-docs diff.
        ("pull_request", (), ("docs/index.md", "docs/user-guide/concepts.md"), True),
        # One non-docs path keeps the fleet.
        ("pull_request", (), ("docs/index.md", "miles/utils/http_utils.py"), False),
        # An empty diff is "unknown", never "nothing relevant changed".
        ("pull_request", (), (), False),
        # Any CI label is an explicit request for tests and wins.
        ("pull_request", ("run-ci-megatron",), ("docs/index.md",), False),
        ("pull_request", ("bypass-fastfail",), ("docs/index.md",), False),
        # Scheduled and manual runs never skip, whatever the list says.
        ("schedule", (), ("docs/index.md",), False),
        ("workflow_dispatch", (), ("docs/index.md",), False),
        # A file literally named "docs" is not the docs tree.
        ("pull_request", (), ("docs",), False),
        # Paths git quoted for non-ASCII characters fail closed.
        ("pull_request", (), ('"docs/\\344\\270\\255.md"',), False),
        # Docs-adjacent sources stay full-run: they are code, not the site tree.
        ("pull_request", (), ("scripts/tools/sync_example_docs.py",), False),
        ("pull_request", (), ("examples/README.md",), False),
    ],
)
def test_docs_only_fails_toward_the_full_fleet(event_name, raw_labels, changed_files, docs_only):
    assert is_docs_only_change(event_name, raw_labels, changed_files) is docs_only


def test_docs_only_flows_through_workflow_policy():
    docs_diff = ("docs/index.md",)
    assert resolve_workflow_inputs("pull_request", "", "[]", docs_diff).docs_only is True
    assert resolve_workflow_inputs("pull_request", "", "[]").docs_only is False
    assert resolve_workflow_inputs("schedule", "0 15 * * 6", "not JSON", docs_diff).docs_only is False
    assert resolve_workflow_inputs("pull_request", "", '["run-ci-megatron"]', docs_diff).docs_only is False


@pytest.mark.parametrize("labels_json", ["{", "{}", "null", '["run-ci-megatron", 1]'])
def test_pull_request_rejects_non_string_array_labels(labels_json):
    with pytest.raises(ValueError, match="PR labels were not a JSON string array"):
        resolve_workflow_inputs("pull_request", "", labels_json)


def test_unknown_schedule_is_not_assumed_to_be_nightly():
    with pytest.raises(ValueError, match=r"No CI policy is defined for schedule: 0 0 \* \* 0"):
        resolve_workflow_inputs("schedule", "0 0 * * 0", "[]")


def test_unknown_trigger_is_rejected():
    with pytest.raises(ValueError, match="Unsupported PR Test trigger: push"):
        resolve_workflow_inputs("push", "", "[]")


def test_nightly_label_and_nightly_schedule_share_the_same_run_policy():
    labeled = resolve_workflow_inputs("pull_request", "", '["nightly"]')
    scheduled = resolve_workflow_inputs("schedule", "0 15 * * 0-5", "not JSON")

    assert resolve_policy(labeled.cadence, set(labeled.raw_labels)) == resolve_policy(
        scheduled.cadence, set(scheduled.raw_labels)
    )


def test_weekly_schedule_resolves_to_independent_full_policy():
    scheduled = resolve_workflow_inputs("schedule", "0 15 * * 6", "not JSON")
    policy = resolve_policy(scheduled.cadence, set(scheduled.raw_labels))

    assert policy.cadence == WEEKLY_CADENCE
    assert policy.admit_nightly_tests is True
    assert policy.bypass_fastfail is True
    assert policy.write_baseline is True


@pytest.mark.parametrize(
    ("labels_json", "expected_outputs"),
    [
        (
            "[]",
            "existing=value\ncadence=regular\nraw_labels=\nbypass_fastfail=false\ndocs_only=false\n",
        ),
        (
            '["run-ci-megatron", "nightly", "run-ci-megatron", "ignored"]',
            "existing=value\ncadence=nightly\n"
            "raw_labels=run-ci-megatron nightly run-ci-megatron\n"
            "bypass_fastfail=true\ndocs_only=false\n",
        ),
    ],
)
def test_cli_appends_exact_github_outputs(tmp_path, labels_json, expected_outputs):
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    output_path.write_text("existing=value\n")
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "pull_request",
            "SCHEDULE": "",
            "PR_LABELS_JSON": labels_json,
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.read_text() == expected_outputs
    assert "Resolved CI policy:" in result.stdout


@pytest.mark.parametrize(
    ("changed_files_text", "docs_only"),
    [
        ("docs/index.md\ndocs/ci/00-stage.md\n", "true"),
        ("docs/index.md\nmiles/train.py\n", "false"),
        ("", "false"),
        (None, "false"),  # CHANGED_FILES_PATH points at a file that does not exist
    ],
)
def test_cli_resolves_docs_only_from_the_change_list_file(tmp_path, changed_files_text, docs_only):
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    changed_path = tmp_path / "changed_files.txt"
    if changed_files_text is not None:
        changed_path.write_text(changed_files_text)
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "pull_request",
            "SCHEDULE": "",
            "PR_LABELS_JSON": "[]",
            "CHANGED_FILES_PATH": str(changed_path),
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"docs_only={docs_only}\n" in output_path.read_text()


def test_cli_fails_for_unknown_schedule(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "schedule",
            "SCHEDULE": "0 0 * * 0",
            "PR_LABELS_JSON": "not JSON",
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "::error::No CI policy is defined for schedule: 0 0 * * 0" in result.stderr
    assert not output_path.exists()

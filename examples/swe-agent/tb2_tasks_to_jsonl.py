#!/usr/bin/env python3
"""Convert a Terminal-Bench-2 harbor task export into a JSONL of prompts.

``harbor datasets download terminal-bench@2.0 --export -o <dir>`` writes one
directory per task (``instruction.md``, ``task.toml``, ``environment/``,
``solution/``, ``tests/``). ``download_and_process_data.py`` only accepts a
``.jsonl`` path or a HuggingFace dataset name, and the ``prepare_harbor_tasks.py``
its docstring points at does not exist in the tree, so nothing currently bridges
the two. This does.

Emits one record per task with the instruction text under ``instruction`` and the
task identity plus environment hints alongside, so a downstream pass can enrich
metadata and the harbor agent server can resolve the task directory.

Usage:
    python tb2_tasks_to_jsonl.py \\
        --tasks_dir /data/home/sdong/datasets/tb2/terminal-bench \\
        --output /data/home/sdong/datasets/tb2/tb2_tasks_raw.jsonl
"""

import json
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

from tap import Tap


class Args(Tap):
    tasks_dir: Path  # Directory holding one sub-directory per TB2 task
    output: Path  # Destination .jsonl
    limit: int | None = None  # Optionally cap the number of tasks emitted


def _read_task(task_dir: Path) -> dict[str, object] | None:
    """Build one JSONL record from a single TB2 task directory."""
    instruction_path = task_dir / "instruction.md"
    toml_path = task_dir / "task.toml"
    if not instruction_path.is_file() or not toml_path.is_file():
        return None

    with toml_path.open("rb") as f:
        spec = tomllib.load(f)

    metadata = spec.get("metadata", {})
    environment = spec.get("environment", {})

    return {
        "task_id": task_dir.name,
        "task_dir": task_dir.name,
        "instruction": instruction_path.read_text(),
        "difficulty": metadata.get("difficulty"),
        "category": metadata.get("category"),
        "tags": metadata.get("tags", []),
        "docker_image": environment.get("docker_image"),
        "agent_timeout_sec": spec.get("agent", {}).get("timeout_sec"),
        "verifier_timeout_sec": spec.get("verifier", {}).get("timeout_sec"),
    }


def _iter_tasks(tasks_dir: Path) -> Iterator[dict[str, object]]:
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        record = _read_task(task_dir)
        if record is None:
            print(f"  skipping {task_dir.name}: missing instruction.md or task.toml")
            continue
        yield record


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()

    if not args.tasks_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {args.tasks_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w") as out:
        for record in _iter_tasks(args.tasks_dir):
            if args.limit is not None and count >= args.limit:
                break
            out.write(json.dumps(record) + "\n")
            count += 1

    print(f"wrote {count} tasks -> {args.output}")


if __name__ == "__main__":
    main()

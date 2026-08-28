"""Validate and profile a conversational SFT dataset with the training tokenizer.

The command streams JSONL row by row (so heterogeneous nested tool schemas are
valid), reads Parquet through Polars, renders every conversation through Miles'
real SFT loss-mask implementation, rejects malformed or overlong rows, and
prints token-length quantiles used to choose the training budget.

Example:
  python scripts/tools/validate_sft_dataset.py \
    --dataset /root/datasets/train.parquet \
    --model /root/models/Qwen3.6-35B-A3B \
    --max-seq-len 65536
"""

import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from tap import Tap

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


class Arguments(Tap):
    dataset: Path
    model: Path
    input_key: str = "messages"
    tools_key: str = ""
    chat_template_path: Path | None = None
    loss_mask_type: Literal["qwen", "qwen3", "distill_qwen"] = "qwen3"
    max_seq_len: int = 65536
    max_errors: int = 20

    def configure(self) -> None:
        self.add_argument(
            "--dataset",
            type=Path,
            help="JSONL or Parquet dataset to validate.",
        )
        self.add_argument(
            "--model",
            type=Path,
            help="Hugging Face checkpoint containing the tokenizer.",
        )
        self.add_argument(
            "--input-key",
            type=str,
            default="messages",
            help="Column containing the conversation messages.",
        )
        self.add_argument(
            "--tools-key",
            type=str,
            default="",
            help="Optional column containing tool definitions.",
        )
        self.add_argument(
            "--chat-template-path",
            type=Path,
            default=None,
            help="Optional chat-template override.",
        )
        self.add_argument(
            "--loss-mask-type",
            type=str,
            choices=("qwen", "qwen3", "distill_qwen"),
            default="qwen3",
            help="Miles SFT loss-mask implementation.",
        )
        self.add_argument(
            "--max-seq-len",
            type=int,
            default=65536,
            help="Reject rendered rows longer than this token count.",
        )
        self.add_argument(
            "--max-errors",
            type=int,
            default=20,
            help="Maximum row errors to print before stopping.",
        )


@dataclass(frozen=True)
class RowStats:
    total_tokens: int
    loss_tokens: int
    response_span_tokens: int


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object] | None, str | None]]:
    row_index = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                yield row_index, None, f"line {line_number}: invalid JSON: {error}"
            else:
                if isinstance(value, dict):
                    yield row_index, value, None
                else:
                    yield row_index, None, f"line {line_number}: row must be a JSON object"
            row_index += 1


def _iter_dataset(path: Path) -> Iterator[tuple[int, dict[str, object] | None, str | None]]:
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        for row_index, row in enumerate(frame.iter_rows(named=True)):
            yield row_index, row, None
        return
    if path.suffix in {".jsonl", ".ndjson"}:
        yield from _iter_jsonl(path)
        return
    raise ValueError(f"Unsupported dataset extension {path.suffix!r}; expected .jsonl, .ndjson, or .parquet")


def _decode_json_value(value: object, *, field: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is a string but not valid JSON: {error}") from error


def _validate_messages(value: object) -> list[dict[str, object]]:
    value = _decode_json_value(value, field="messages")
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")

    messages: list[dict[str, object]] = []
    has_nonempty_assistant = False
    for message_index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{message_index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{message_index}].role must be a non-empty string")
        if not isinstance(content, (str, list)):
            raise ValueError(f"messages[{message_index}].content must be a string or content-block list")
        if role == "assistant" and bool(content):
            has_nonempty_assistant = True
        messages.append(message)

    if not has_nonempty_assistant:
        raise ValueError("messages must contain at least one non-empty assistant turn")
    return messages


def _validate_tools(value: object) -> list[dict[str, object]] | None:
    if value is None:
        return None
    value = _decode_json_value(value, field="tools")
    if not isinstance(value, list):
        raise ValueError("tools must be a list when present")
    if not all(isinstance(tool, dict) for tool in value):
        raise ValueError("every tools entry must be an object")
    return value


def _validate_row(
    row: dict[str, object],
    *,
    input_key: str,
    tools_key: str,
    max_seq_len: int,
    mask_generator: MultiTurnLossMaskGenerator,
) -> RowStats:
    if input_key not in row:
        raise ValueError(f"missing input column {input_key!r}")
    messages = _validate_messages(row[input_key])
    tools = _validate_tools(row.get(tools_key)) if tools_key else None
    token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
    if len(token_ids) != len(loss_mask):
        raise ValueError(f"token/mask length mismatch: {len(token_ids)} != {len(loss_mask)}")
    if len(token_ids) > max_seq_len:
        raise ValueError(f"rendered length {len(token_ids)} exceeds max_seq_len={max_seq_len}")
    loss_tokens = sum(loss_mask)
    if loss_tokens == 0:
        raise ValueError("conversation produces zero assistant loss tokens")
    response_span = mask_generator.get_response_lengths([loss_mask])[0]
    return RowStats(total_tokens=len(token_ids), loss_tokens=loss_tokens, response_span_tokens=response_span)


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {name: None for name in ("min", "p50", "p90", "p95", "p99", "max")}
    series = pl.Series("value", values)
    return {
        "min": series.min(),
        "p50": series.quantile(0.50, interpolation="nearest"),
        "p90": series.quantile(0.90, interpolation="nearest"),
        "p95": series.quantile(0.95, interpolation="nearest"),
        "p99": series.quantile(0.99, interpolation="nearest"),
        "max": series.max(),
    }


def _summary(
    dataset: Path,
    *,
    rows_seen: int,
    schema: dict[str, set[str]],
    stats: list[RowStats],
    max_seq_len: int,
    error_count: int,
) -> dict[str, object]:
    total_token_values = [item.total_tokens for item in stats]
    length_thresholds = (8192, 16384, 32768, 65536, 131072, 262144)
    return {
        "dataset": str(dataset.resolve()),
        "rows_seen": rows_seen,
        "validated_rows": len(stats),
        "error_count": error_count,
        "max_seq_len": max_seq_len,
        "schema": {name: sorted(types) for name, types in sorted(schema.items())},
        "rows_above_token_threshold": {
            str(threshold): sum(value > threshold for value in total_token_values) for threshold in length_thresholds
        },
        "total_tokens": _quantiles(total_token_values),
        "loss_tokens": _quantiles([item.loss_tokens for item in stats]),
        "response_span_tokens": _quantiles([item.response_span_tokens for item in stats]),
        "totals": asdict(
            RowStats(
                total_tokens=sum(item.total_tokens for item in stats),
                loss_tokens=sum(item.loss_tokens for item in stats),
                response_span_tokens=sum(item.response_span_tokens for item in stats),
            )
        ),
    }


def main() -> None:
    args = Arguments().parse_args()
    if args.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if args.max_errors <= 0:
        raise ValueError("max_errors must be positive")

    tokenizer = load_tokenizer(
        str(args.model),
        chat_template_path=str(args.chat_template_path) if args.chat_template_path is not None else None,
        trust_remote_code=True,
    )
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)

    stats: list[RowStats] = []
    errors: list[str] = []
    schema: dict[str, set[str]] = {}
    rows_seen = 0
    for row_index, row, read_error in _iter_dataset(args.dataset):
        rows_seen += 1
        if read_error is not None:
            errors.append(f"row {row_index}: {read_error}")
            if len(errors) >= args.max_errors:
                break
            continue

        assert row is not None
        for key, value in row.items():
            schema.setdefault(key, set()).add(type(value).__name__)
        try:
            stats.append(
                _validate_row(
                    row,
                    input_key=args.input_key,
                    tools_key=args.tools_key,
                    max_seq_len=args.max_seq_len,
                    mask_generator=mask_generator,
                )
            )
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            errors.append(f"row {row_index}: {error}")
            if len(errors) >= args.max_errors:
                break

    if rows_seen == 0:
        raise ValueError("dataset is empty")
    print(
        json.dumps(
            _summary(
                args.dataset,
                rows_seen=rows_seen,
                schema=schema,
                stats=stats,
                max_seq_len=args.max_seq_len,
                error_count=len(errors),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        print("SFT dataset validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

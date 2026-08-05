from typing import NamedTuple


class ParsedCellId(NamedTuple):
    spec_name: str
    cell_index: int


def parse_cell_id(cell_id: str) -> ParsedCellId:
    spec_name, cell_index = cell_id.rsplit("-", maxsplit=1)
    return ParsedCellId(spec_name=spec_name, cell_index=int(cell_index))


def compute_cell_id(*, spec_name: str, cell_index: int) -> str:
    return f"{spec_name}-{cell_index}"


# TODO refactor & move later
def compute_worker_name(*, spec_name: str, cell_index: int = 0, worker_in_cell_index: int = 0) -> str:
    return f"{spec_name}-{cell_index}-{worker_in_cell_index}"


# TODO refactor & move later
def parse_worker_name(worker_name: str) -> tuple[str, int, int]:
    spec_name, cell_index, worker_in_cell_index = worker_name.rsplit("-", maxsplit=2)
    return spec_name, int(cell_index), int(worker_in_cell_index)


CHART_NAME = "miles-run"
NAME_BUDGET = 52


def fullname(release: str, chart_name: str = CHART_NAME) -> str:
    name = release if chart_name in release else f"{release}-{chart_name}"
    return _trim_suffix(_trunc(name, NAME_BUDGET), "-")


def component_name(release: str, component: str, chart_name: str = CHART_NAME) -> str:
    budget = NAME_BUDGET - (len(component) + 1)
    prefix = _trim_suffix(_trunc(fullname(release, chart_name), budget), "-")
    return f"{prefix}-{component}"


def static_worker_host(release: str, component: str, cell_index: int = 0) -> str:
    name = component_name(release, component)
    return f"{name}-{cell_index}.{name}"


def _trunc(value: str, count: int) -> str:
    return value[count:] if count < 0 else value[:count]


def _trim_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value

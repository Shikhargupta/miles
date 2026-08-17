from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import pydantic

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.naming import cell_id_of_worker, parse_cell_id
from miles.utils.workers.reconcile.list_based import ListBasedReconcileLoop
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)

REGISTERED_CELLS_POLL_INTERVAL_SECONDS = 5.0


@dataclass(kw_only=True)
class RegistrationHub(BaseWorkerProvider):
    expected_num_reporters: int
    _state_of_reporter_id: dict[str, _ReporterState] = field(init=False, default_factory=dict)
    _cell_of_cell_id: dict[str, RegisteredCellInfo] = field(init=False, default_factory=dict)
    _watched: bool = field(init=False, default=False)

    # ========================== Taking in snapshots ===========================

    async def ingest(self, snapshot: RegistrationSnapshot) -> None:
        _assert_every_cell_is_addressable(snapshot)
        cell_of_cell_id = {cell.info.cell_id: cell for cell in snapshot.cells}

        held = self._state_of_reporter_id.get(snapshot.reporter_id)
        if held is not None and snapshot.sequence_number <= held.sequence_number:
            logger.warning(
                f"Ignoring snapshot {snapshot.sequence_number} of reporter {snapshot.reporter_id}: snapshot "
                f"{held.sequence_number} is at least as new, so this one arrived late"
            )
            return

        state = self._state_of_reporter_id.setdefault(snapshot.reporter_id, _ReporterState())
        self._replace_membership(reporter_id=snapshot.reporter_id, cell_of_cell_id=cell_of_cell_id)
        state.sequence_number = snapshot.sequence_number
        state.expected_num_cells_by_group_id = dict(snapshot.expected_num_cells_by_group_id)

    def _replace_membership(self, *, reporter_id: str, cell_of_cell_id: dict[str, RegisteredCellInfo]) -> None:
        for cell_id in cell_of_cell_id:
            assert (owner := self._cell_of_cell_id.get(cell_id)) is None or owner.reporter_id == reporter_id, (
                f"reporter {reporter_id} reports cell {cell_id}, but reporter {owner.reporter_id} already reported "
                f"it, so two deployments of this run share a pool id"
            )

        held = {cell_id for cell_id, cell in self._cell_of_cell_id.items() if cell.reporter_id == reporter_id}
        for cell_id in held - set(cell_of_cell_id):
            del self._cell_of_cell_id[cell_id]
        self._cell_of_cell_id.update(cell_of_cell_id)

    # ======================= Serving the reported cells =======================

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        assert not self._watched, "a registration hub reports to exactly one watcher"
        self._watched = True
        loop = ListBasedReconcileLoop(
            list_cells=self._list_cells,
            poll_interval_seconds=REGISTERED_CELLS_POLL_INTERVAL_SECONDS,
        )
        return await loop.start(reconcile)

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        cell = self._cell_of_cell_id[cell_id_of_worker(worker_name)]
        worker = next((one for one in cell.workers if one.name == worker_name), None)
        assert worker is not None, (
            f"{worker_name} is not one of the workers {sorted(one.name for one in cell.workers)} that reporter "
            f"{cell.reporter_id} reported for cell {cell.info.cell_id}"
        )
        return worker.self_addrs

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._worker_infos_of_cell(cell_id) for cell_id in cell_ids]

    def extra_expected_num_cells(self, *, group_id: str) -> int:
        reported = self._state_of_reporter_id
        assert len(reported) >= self.expected_num_reporters, (
            f"{len(reported)}/{self.expected_num_reporters} reporters have reported themselves "
            f"({sorted(reported)}), so the cells of the missing ones are not known yet"
        )
        return sum(state.expected_num_cells_by_group_id.get(group_id, 0) for state in reported.values())

    def _cell_ids(self) -> list[str]:
        return sorted(self._cell_of_cell_id)

    async def _list_cells(self) -> dict[str, CellInfo]:
        return {cell_id: cell.info for cell_id, cell in self._cell_of_cell_id.items()}

    def _worker_infos_of_cell(self, cell_id: str) -> list[WorkerInfo]:
        return self._cell_of_cell_id[cell_id].workers


class _ReporterState(StrictBaseModel):
    sequence_number: int = -1
    expected_num_cells_by_group_id: dict[str, int] = pydantic.Field(default_factory=dict)


def _assert_every_cell_is_addressable(snapshot: RegistrationSnapshot) -> None:
    occurrences = Counter(cell.info.cell_id for cell in snapshot.cells)
    for cell in snapshot.cells:
        assert cell.reporter_id == snapshot.reporter_id, (
            f"the snapshot of reporter {snapshot.reporter_id} carries cell {cell.info.cell_id} of reporter "
            f"{cell.reporter_id}, so the snapshot was assembled from the membership of two deployments"
        )
        assert occurrences[cell.info.cell_id] == 1, (
            f"reporter {snapshot.reporter_id} carries cell {cell.info.cell_id} {occurrences[cell.info.cell_id]} "
            f"times in one snapshot, and a cell id names exactly one cell, so either entry would hold a membership "
            f"the reporter never announced"
        )
        _assert_cell_is_addressable(cell)


def _assert_cell_is_addressable(cell: RegisteredCellInfo) -> None:
    prefix = f"reporter {cell.reporter_id} reports cell {cell.info.cell_id}, which this run cannot take in:"
    try:
        pool_id = parse_cell_id(cell.info.cell_id).pool_id
        worker_cell_ids = {cell_id_of_worker(worker.name) for worker in cell.workers}
    except ValueError as cause:
        raise AssertionError(
            f"{prefix} its cell id, or the name of one of its workers, does not read as <pool id>-<cell index>, and "
            f"this run parses those names to address the workers of that cell"
        ) from cause

    assert pool_id == cell.info.pool_id, (
        f"{prefix} it does not name its own pool {cell.info.pool_id}, and a reporter namespaces its pool ids so "
        f"that two deployments never collide"
    )
    assert cell.workers, f"{prefix} it carries no worker to address"
    assert worker_cell_ids == {
        cell.info.cell_id
    }, f"{prefix} its workers {sorted(one.name for one in cell.workers)} do not all belong to it"

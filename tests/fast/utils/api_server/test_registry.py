from __future__ import annotations

import pytest

from miles.utils.ft_utils.api_server.registry import _CellRegistry

from .conftest import MockHandler


class TestListCells:
    async def test_lists_the_cells_of_every_handler(self) -> None:
        """One kind of cell must not hide another from the ft controller."""
        actor, rollout = MockHandler("actor"), MockHandler("rollout")
        actor.add("0")
        rollout.add("1")
        registry = _CellRegistry([actor, rollout])

        assert [cell.metadata.name for cell in await registry.list_cells()] == ["actor-0", "rollout-1"]

    async def test_a_handler_without_cells_contributes_nothing(self) -> None:
        """A train-only or rollout-only deployment still answers the same endpoint."""
        registry = _CellRegistry([MockHandler("actor"), MockHandler("rollout")])

        assert await registry.list_cells() == []

    async def test_cells_added_after_construction_are_listed(self) -> None:
        """Cells come and go while the server runs, so the set is resolved per request."""
        rollout = MockHandler("rollout")
        registry = _CellRegistry([rollout])
        assert await registry.list_cells() == []

        rollout.add("0")

        assert [cell.metadata.name for cell in await registry.list_cells()] == ["rollout-0"]

    async def test_cells_removed_after_construction_disappear(self) -> None:
        """A cell that was scaled away must stop being reported as existing."""
        rollout = MockHandler("rollout")
        rollout.add("0")
        registry = _CellRegistry([rollout])

        del rollout.cells["0"]

        assert await registry.list_cells() == []


class TestResolve:
    async def test_a_name_resolves_to_its_handler_and_key(self) -> None:
        """The api name carries the cell type, which is what picks the handler."""
        actor, rollout = MockHandler("actor"), MockHandler("rollout")
        rollout.add("inference-engine-0-0-2")
        registry = _CellRegistry([actor, rollout])

        handler, cell_key = await registry.resolve("rollout-inference-engine-0-0-2")

        assert handler is rollout
        assert cell_key == "inference-engine-0-0-2"

    async def test_an_unknown_name_raises(self) -> None:
        """An unknown cell must 404 rather than resolve to a neighbour."""
        registry = _CellRegistry([MockHandler("actor")])

        with pytest.raises(KeyError):
            await registry.resolve("actor-7")

    async def test_a_name_with_an_unknown_type_raises(self) -> None:
        """Only registered kinds of cells are addressable."""
        registry = _CellRegistry([MockHandler("actor")])

        with pytest.raises(KeyError):
            await registry.resolve("critic-0")

    async def test_a_cell_key_containing_dashes_survives_the_round_trip(self) -> None:
        """Engine cell ids contain dashes, so only the type prefix may be stripped."""
        rollout = MockHandler("rollout")
        rollout.add("inference-engine-0-1-3")
        registry = _CellRegistry([rollout])

        _handler, cell_key = await registry.resolve("rollout-inference-engine-0-1-3")

        assert cell_key == "inference-engine-0-1-3"

    def test_duplicate_cell_types_are_rejected(self) -> None:
        """Two handlers of one type would make cell names ambiguous."""
        with pytest.raises(AssertionError, match="Duplicate cell types"):
            _CellRegistry([MockHandler("rollout"), MockHandler("rollout")])

import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo


def _make_cell_info(**overrides) -> CellInfo:
    kwargs = dict(cell_id="cell-0", spec_name="engine", members_hash="hash-0", member_urls=["http://host-0:8000"])
    kwargs.update(overrides)
    return CellInfo(**kwargs)


class TestCellInfo:
    def test_constructs_and_exposes_fields(self):
        """A cell info keeps its id, members hash, and member urls as provided."""
        cell_info = _make_cell_info(member_urls=["http://host-0:8000", "http://host-1:8000"])
        assert cell_info.cell_id == "cell-0"
        assert cell_info.spec_name == "engine"
        assert cell_info.members_hash == "hash-0"
        assert cell_info.member_urls == ["http://host-0:8000", "http://host-1:8000"]

    def test_rejects_extra_field(self):
        """Unknown fields are forbidden."""
        with pytest.raises(ValidationError):
            _make_cell_info(unknown_field=1)

    def test_is_frozen(self):
        """Field assignment after construction is rejected."""
        cell_info = _make_cell_info()
        with pytest.raises(ValidationError):
            cell_info.members_hash = "hash-1"


class TestBaseWorkerProvider:
    def test_cannot_instantiate_abstract_contract(self):
        """The provider contract itself cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseWorkerProvider()

    def test_incomplete_implementation_rejected(self):
        """A provider missing watch_cells cannot be instantiated."""

        class Incomplete(BaseWorkerProvider):
            def get_handle(self, worker_name: str):
                raise NotImplementedError

            def get_url(self, worker_name: str) -> str:
                raise NotImplementedError

        with pytest.raises(TypeError):
            Incomplete()

from __future__ import annotations

import pytest
from tests.fast.utils.workers.reconcile.utils import make_pod, pod_cell

from miles.utils.workers.reconcile.object_store import ObjectStore
from miles.utils.workers.reconcile.source_event import Delete, SyncDone, SyncStart, Upsert


def make_store() -> ObjectStore:
    return ObjectStore(key_map=pod_cell)


class TestIncrementalEvents:
    def test_an_upsert_stores_the_object_and_reports_its_parent(self):
        """A plain upsert lands in the store and wakes exactly its cell."""
        store = make_store()
        update = store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))

        assert update.affected_parents == {"cell-a"}
        assert "pod-0" in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-0"]

    def test_a_reparenting_upsert_reports_both_parents(self):
        """Moving an object between cells affects the old and the new one."""
        store = make_store()
        store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        update = store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell="cell-b")))

        assert update.affected_parents == {"cell-a", "cell-b"}
        assert store.get_by_parent("cell-a") == []

    def test_a_delete_reports_the_stored_parent(self):
        """Deleting a known object affects the cell it belonged to."""
        store = make_store()
        store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        update = store.handle_event(Delete(key="pod-0", last_obj=None))

        assert update.affected_parents == {"cell-a"}
        assert "pod-0" not in store

    def test_a_delete_of_an_unknown_object_uses_the_tombstone(self):
        """An unknown delete is attributed through last_obj."""
        store = make_store()
        update = store.handle_event(Delete(key="pod-0", last_obj=make_pod("pod-0", cell="cell-a")))

        assert update.affected_parents == {"cell-a"}

    def test_an_unmappable_upsert_is_dropped_and_removes_any_stored_object(self):
        """A key_map failure turns the upsert into a departure, not a stale entry."""
        store = make_store()
        store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        update = store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell=None)))

        assert update.affected_parents == {"cell-a"}
        assert "pod-0" not in store


class TestSegments:
    def test_a_segment_buffers_events_and_replaces_on_sync_done(self):
        """Upserts inside SyncStart/SyncDone apply atomically as a store replace."""
        store = make_store()
        store.handle_event(Upsert(key="pod-old", obj=make_pod("pod-old", cell="cell-a")))

        assert store.handle_event(SyncStart()).affected_parents == set()
        assert store.handle_event(Upsert(key="pod-new", obj=make_pod("pod-new", cell="cell-b"))).affected_parents == set()
        update = store.handle_event(SyncDone())

        assert update.affected_parents == {"cell-a", "cell-b"}
        assert "pod-old" not in store
        assert [pod.metadata.name for pod in store.get_by_parent("cell-b")] == ["pod-new"]

    def test_a_delete_inside_a_segment_removes_the_buffered_upsert(self):
        """A delete arriving mid-segment must not survive into the replace."""
        store = make_store()
        store.handle_event(SyncStart())
        store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        store.handle_event(Delete(key="pod-0", last_obj=None))
        store.handle_event(SyncDone())

        assert "pod-0" not in store

    def test_reset_segment_discards_a_partial_listing(self):
        """A reopened stream must not leak the previous half-received segment."""
        store = make_store()
        store.handle_event(SyncStart())
        store.handle_event(Upsert(key="pod-0", obj=make_pod("pod-0", cell="cell-a")))
        store.reset_segment()

        assert store.handle_event(SyncStart()).affected_parents == set()
        store.handle_event(SyncDone())
        assert "pod-0" not in store

    def test_unpaired_markers_raise(self):
        """SyncStart inside a segment and SyncDone outside one violate the contract."""
        store = make_store()
        with pytest.raises(RuntimeError):
            store.handle_event(SyncDone())
        store.handle_event(SyncStart())
        with pytest.raises(RuntimeError):
            store.handle_event(SyncStart())


class TestQueries:
    def test_get_by_parent_returns_members_sorted_by_key(self):
        """Membership listing is deterministic."""
        store = make_store()
        store.handle_event(Upsert(key="pod-b", obj=make_pod("pod-b", cell="cell-a")))
        store.handle_event(Upsert(key="pod-a", obj=make_pod("pod-a", cell="cell-a")))

        assert [pod.metadata.name for pod in store.get_by_parent("cell-a")] == ["pod-a", "pod-b"]

    def test_parent_keys_lists_every_cell_that_still_has_members(self):
        """parent_keys is what a resync re-drives."""
        store = make_store()
        store.handle_event(Upsert(key="pod-a", obj=make_pod("pod-a", cell="cell-a")))
        store.handle_event(Upsert(key="pod-b", obj=make_pod("pod-b", cell="cell-b")))
        store.handle_event(Delete(key="pod-b", last_obj=None))

        assert store.parent_keys() == {"cell-a"}

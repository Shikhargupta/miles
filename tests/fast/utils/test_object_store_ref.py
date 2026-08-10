from __future__ import annotations

import pytest

from miles.utils.object_store import ObjectStoreBackend, StoreObjectRef


class TestStoreObjectRef:
    def test_a_mooncake_reference_survives_a_json_round_trip(self):
        """The trainer ships it to the driver over rpc, which encodes the model as json."""
        ref = StoreObjectRef(backend=ObjectStoreBackend.MOONCAKE, payload={"key": "miles-object-store/7", "size": 12})

        restored = StoreObjectRef.model_validate_json(ref.model_dump_json())

        assert restored == ref
        assert restored.payload == {"key": "miles-object-store/7", "size": 12}

    def test_a_reference_says_which_store_can_redeem_it(self):
        """Two stores may be alive in one run, and redeeming a ref in the wrong one returns another object."""
        assert StoreObjectRef(backend=ObjectStoreBackend.RAY, payload=object()).backend == ObjectStoreBackend.RAY

    def test_a_reference_cannot_be_repointed_after_it_is_handed_over(self):
        """A consumer holding it must read the object the producer put, not one somebody swapped in."""
        ref = StoreObjectRef(backend=ObjectStoreBackend.MOONCAKE, payload="k")

        with pytest.raises(Exception):
            ref.payload = "other"

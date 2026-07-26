import pytest

from miles.utils.run_uuid import generate_run_uuid
from miles.utils.weight_version import WeightVersion, try_parse_weight_version_rollout_id

RUN_UUID = "ab12cd34ef5678ab"


class TestWeightVersion:
    def test_serialize_deserialize_roundtrip(self):
        """A serialized weight version decodes back to the same run uuid and rollout id."""
        version = WeightVersion(run_uuid=RUN_UUID, rollout_id=42)
        assert WeightVersion.deserialize(version.serialize()) == version

    def test_serializes_to_the_exact_wire_string(self):
        """Pinning the wire form catches a serializer and parser that drift together."""
        assert WeightVersion(run_uuid=RUN_UUID, rollout_id=7).serialize() == "ab12cd34ef5678ab-00000007"

    def test_the_launch_run_uuid_survives_a_roundtrip(self):
        """The weight version embeds whatever the run identity produced, so the two formats must agree."""
        run_uuid = generate_run_uuid()
        assert (
            WeightVersion.deserialize(WeightVersion(run_uuid=run_uuid, rollout_id=0).serialize()).run_uuid == run_uuid
        )

    @pytest.mark.parametrize(
        "run_uuid, rollout_id",
        [
            ("AB12CD34EF5678AB", 7),
            ("ab12cd34", 7),
            ("ab12cd34ef5678abcd", 7),
            ("ab12cd34ef5678ab", 10**8),
            ("ab12cd34ef5678ab", -1),
        ],
    )
    def test_serializing_an_unrepresentable_version_fails_loudly(self, run_uuid, rollout_id):
        """A version that cannot be read back would silently stamp samples with an unattributable string."""
        with pytest.raises((AssertionError, ValueError)):
            WeightVersion(run_uuid=run_uuid, rollout_id=rollout_id).serialize()

    @pytest.mark.parametrize("bad", ["default", "0", "1", "", "ab12cd34ef5678ab-1", "ab12cd34ef5678ab:00000001", None])
    def test_deserialize_rejects_legacy_and_malformed_versions(self, bad):
        """Bare counters, sglang defaults, and malformed strings are not valid weight versions."""
        with pytest.raises(ValueError, match="invalid weight version"):
            WeightVersion.deserialize(bad)

    def test_parse_rollout_id_rejects_anything_but_the_run_scoped_format(self):
        """A bare counter from an older run is not comparable to a run-scoped id, so it is not one."""
        assert try_parse_weight_version_rollout_id(WeightVersion(run_uuid=RUN_UUID, rollout_id=7).serialize()) == 7
        assert try_parse_weight_version_rollout_id("12") is None
        assert try_parse_weight_version_rollout_id("default") is None

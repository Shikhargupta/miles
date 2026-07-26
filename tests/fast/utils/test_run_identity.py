import pytest

from miles.utils.run_identity import RUN_UUID_LENGTH, generate_run_uuid, validate_run_uuid


class TestGenerateRunUuid:
    def test_generated_uuid_is_accepted_by_the_validator(self):
        """What we generate must be what we accept, or an auto-generated launch fails validation."""
        for _ in range(100):
            assert validate_run_uuid(generate_run_uuid())

    def test_generated_uuid_is_exactly_the_pinned_length(self):
        """A longer or shorter uuid silently changes every string that embeds it."""
        assert len(generate_run_uuid()) == RUN_UUID_LENGTH

    def test_two_launches_get_different_uuids(self):
        """The whole point is that two runs never share one, unlike a human-readable run name."""
        assert len({generate_run_uuid() for _ in range(100)}) == 100


class TestValidateRunUuid:
    def test_accepts_a_well_formed_uuid_and_returns_it(self):
        """The validator is used inline in an assignment, so it must pass the value through."""
        assert validate_run_uuid("ab12cd34") == "ab12cd34"

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "ab12cd3",
            "ab12cd345",
            "AB12CD34",
            "ab12cd3g",
            "ab12-cd3",
            " ab12cd34",
            "ab12cd34 ",
            "my-experiment",
        ],
    )
    def test_rejects_anything_that_is_not_eight_lowercase_hex_characters(self, bad):
        """A user-supplied uuid is rejected at startup rather than corrupting strings later."""
        with pytest.raises(ValueError, match="invalid run uuid"):
            validate_run_uuid(bad)

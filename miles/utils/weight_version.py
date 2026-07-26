import re
from dataclasses import dataclass

from miles.utils.run_uuid import RUN_UUID_LENGTH

_ROLLOUT_ID_DIGITS = 8
_SERIALIZED_PATTERN = re.compile(
    rf"^(?P<run_uuid>[0-9a-f]{{{RUN_UUID_LENGTH}}})-(?P<rollout_id>[0-9]{{{_ROLLOUT_ID_DIGITS}}})$"
)


@dataclass(frozen=True)
class WeightVersion:
    run_uuid: str
    # TODO rollout_id identifies the weights only while updates stay 1:1 with rollouts;
    # switch to a train weight id once they can diverge.
    rollout_id: int

    def serialize(self) -> str:
        result = f"{self.run_uuid}-{self.rollout_id:0{_ROLLOUT_ID_DIGITS}d}"
        assert WeightVersion.deserialize(result) == self
        return result

    @staticmethod
    def deserialize(value: str) -> "WeightVersion":
        match = _SERIALIZED_PATTERN.match(str(value))
        if match is None:
            raise ValueError(
                f"invalid weight version {value!r}; "
                f"expected '<{RUN_UUID_LENGTH}-hex-run-uuid>-<{_ROLLOUT_ID_DIGITS}-digit-rollout-id>'"
            )
        return WeightVersion(run_uuid=match.group("run_uuid"), rollout_id=int(match.group("rollout_id")))


def try_parse_weight_version_rollout_id(value: str) -> int | None:
    try:
        return WeightVersion.deserialize(value).rollout_id
    except ValueError:
        return None

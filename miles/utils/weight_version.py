import re
import uuid
from dataclasses import dataclass

_SERIALIZED_PATTERN = re.compile(r"^(?P<run_uuid>[0-9a-f]{8})-(?P<rollout_id>[0-9]{8})$")


@dataclass(frozen=True)
class WeightVersion:
    run_uuid: str
    # TODO rollout_id identifies the weights only while updates stay 1:1 with rollouts;
    # switch to a train weight id once they can diverge.
    rollout_id: int

    def serialize(self) -> str:
        result = f"{self.run_uuid}-{self.rollout_id:08d}"
        assert _SERIALIZED_PATTERN.match(result), f"malformed weight version {result!r} from {self!r}"
        return result

    @staticmethod
    def deserialize(value: str) -> "WeightVersion":
        match = _SERIALIZED_PATTERN.match(str(value))
        if match is None:
            raise ValueError(f"invalid weight version {value!r}; expected '<8-hex-run-uuid>-<8-digit-rollout-id>'")
        return WeightVersion(run_uuid=match.group("run_uuid"), rollout_id=int(match.group("rollout_id")))


def generate_weight_version_run_uuid() -> str:
    return uuid.uuid4().hex[:8]


def parse_weight_version_rollout_id(value: str) -> int | None:
    try:
        return WeightVersion.deserialize(value).rollout_id
    except ValueError:
        return None

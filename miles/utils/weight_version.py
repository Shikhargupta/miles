import re
from dataclasses import dataclass

from miles.utils.run_uuid import RUN_UUID_LENGTH

_NUM_TRAINED_ROLLOUTS_DIGITS = 8
_SERIALIZED_PATTERN = re.compile(
    rf"(?P<run_uuid>[0-9a-f]{{{RUN_UUID_LENGTH}}})-(?P<num_trained_rollouts>[0-9]{{{_NUM_TRAINED_ROLLOUTS_DIGITS}}})"
)


@dataclass(frozen=True)
class WeightVersion:
    run_uuid: str
    num_trained_rollouts: int

    def serialize(self) -> str:
        result = f"{self.run_uuid}-{self.num_trained_rollouts:0{_NUM_TRAINED_ROLLOUTS_DIGITS}d}"
        assert WeightVersion.deserialize(result) == self
        return result

    @staticmethod
    def deserialize(value: str) -> "WeightVersion":
        match = _SERIALIZED_PATTERN.fullmatch(str(value))
        if match is None:
            raise ValueError(
                f"invalid weight version {value!r}; "
                f"expected '<{RUN_UUID_LENGTH}-hex-run-uuid>-<{_NUM_TRAINED_ROLLOUTS_DIGITS}-digit-trained-rollout-count>'"
            )
        return WeightVersion(
            run_uuid=match.group("run_uuid"), num_trained_rollouts=int(match.group("num_trained_rollouts"))
        )


def try_parse_num_trained_rollouts(value: str) -> int | None:
    try:
        return WeightVersion.deserialize(value).num_trained_rollouts
    except ValueError:
        return None

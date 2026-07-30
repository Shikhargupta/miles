import ray
from pydantic import ConfigDict

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class AddrInfo(FrozenStrictBaseModel):
    server_url: str
    bootstrap_port: int | None = None


# ------------------------- states -----------------------------


class StateBase(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class StateAllocatedBase(StateBase):
    actor_handles: list[ray.actor.ActorHandle]
    addr_infos: list[AddrInfo]


class StateAllocatedUninitialized(StateAllocatedBase):
    pass


class StateAllocatedAlive(StateAllocatedBase):
    pass


CellState = StateAllocatedUninitialized | StateAllocatedAlive

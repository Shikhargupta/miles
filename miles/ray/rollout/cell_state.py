import ray
from pydantic import BaseModel, ConfigDict


class AddrInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_url: str
    bootstrap_port: int | None = None


# ------------------------- states -----------------------------


class StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class StateStopped(StateBase):
    pass


class StateAllocatedBase(StateBase):
    actor_handle: ray.actor.ActorHandle
    addr_info: AddrInfo | None = None


class StateAllocatedUninitialized(StateAllocatedBase):
    pass


class StateAllocatedAlive(StateAllocatedBase):
    pass


CellState = StateStopped | StateAllocatedUninitialized | StateAllocatedAlive

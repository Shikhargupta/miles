import abc


class BaseWorkerManager(abc.ABC):
    """Owns worker lifetime for one infrastructure layer.

    Only the ops boundary commands this; consumers observe workers through a
    ``BaseWorkerProvider`` instead.
    """

    @abc.abstractmethod
    async def start_cell(self, cell_id: str) -> None: ...

    @abc.abstractmethod
    async def stop_cell(self, cell_id: str) -> None: ...

    @abc.abstractmethod
    def cell_ids(self) -> list[str]: ...

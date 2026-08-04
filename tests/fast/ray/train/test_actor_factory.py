from unittest.mock import MagicMock, patch

from miles.ray.train import actor_factory


class _RecordingRemote:
    def __init__(self, record: list, name: str, result=None):
        self._record = record
        self._name = name
        self._result = result

    def remote(self, *args, **kwargs):
        self._record.append((self._name, args, kwargs))
        return self._result


class _RecordingActor:
    def __init__(self, record: list, rank: int):
        self._record = record
        self.rank = rank
        self.propose_master_addr_and_port = _RecordingRemote(record, f"propose-{rank}", ("10.0.0.1", 20001))
        self.configure_master_addr_and_port = _RecordingRemote(record, f"configure-{rank}")


class TestMasterAddrIsAssignedByTheDriver:
    def _allocate(self, record: list, world_size: int) -> list:
        actors = iter([_RecordingActor(record, rank) for rank in range(world_size)])

        def _create(*args, **kwargs):
            actor = next(actors)
            record.append((f"create-{actor.rank}", args, kwargs))
            return actor

        actor_class = MagicMock()
        actor_class.options.return_value.remote.side_effect = _create
        args = MagicMock()
        args.train_backend = "megatron"
        args.use_fault_tolerance = False
        args.offload_train = False

        with (
            patch.object(actor_factory.ray, "remote", return_value=actor_class),
            patch.object(actor_factory.ray, "get", side_effect=lambda x: x),
            patch.object(actor_factory, "PlacementGroupSchedulingStrategy", MagicMock()),
        ):
            return actor_factory.allocate_gpus_for_actor(
                args=args,
                gpus_per_cell=world_size,
                pg=(MagicMock(), list(range(world_size)), list(range(world_size))),
                num_gpus_per_actor=0.4,
                indep_dp_store_addr=None,
                role="actor",
                cell_index=0,
            )

    def test_no_worker_is_created_after_the_master_addr_is_known(self):
        """Asking rank 0 mid-loop is what used to serialize worker creation."""
        record: list = []

        self._allocate(record, world_size=3)

        first_propose = next(i for i, (name, _, _) in enumerate(record) if name.startswith("propose"))
        assert [name for name, _, _ in record[:first_propose]] == ["create-0", "create-1", "create-2"]

    def test_every_worker_is_told_the_same_master_addr(self):
        """All ranks must join the one process group rank 0 proposed."""
        record: list = []

        self._allocate(record, world_size=3)

        configures = [kwargs for name, _, kwargs in record if name.startswith("configure")]
        assert len(configures) == 3
        assert {(c["master_addr"], c["master_port"]) for c in configures} == {("10.0.0.1", 20001)}

import os


class DummyServeWorker:
    def __init__(self, *, tag: str) -> None:
        self._tag = tag
        self._addr_ports: dict = {}

    def configure_addrs_and_ports(self, **kwargs) -> None:
        self._addr_ports = kwargs

    def describe(self) -> dict:
        return {
            "tag": self._tag,
            "addr_ports": self._addr_ports,
            "dummy_env": os.environ.get("MANAGER_DUMMY_ENV"),
        }

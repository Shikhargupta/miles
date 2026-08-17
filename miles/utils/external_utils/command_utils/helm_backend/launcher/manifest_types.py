from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import STATE_FILE_FLAG
from miles.utils.pydantic_utils import FrozenOpenBaseModel, FrozenStrictBaseModel


class EnvEntry(FrozenOpenBaseModel):
    name: str
    value: str = ""


class Container(FrozenOpenBaseModel):
    name: str
    command: list[str] = []
    env: list[EnvEntry] = []


class PodSpec(FrozenOpenBaseModel):
    containers: list[Container] = []


class PodTemplate(FrozenOpenBaseModel):
    spec: PodSpec | None = None


class ObjectSpec(FrozenOpenBaseModel):
    replicas: int | None = None
    template: PodTemplate | None = None


class ObjectMetadata(FrozenOpenBaseModel):
    name: str


class ManifestObject(FrozenOpenBaseModel):
    kind: str
    metadata: ObjectMetadata
    spec: ObjectSpec | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.metadata.name}"

    @property
    def replicas(self) -> int | None:
        return self.spec.replicas if self.spec is not None else None

    @property
    def body(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    @property
    def containers(self) -> list[Container]:
        if self.spec is None or self.spec.template is None:
            return []
        pod = self.spec.template.spec
        return list(pod.containers) if pod is not None else []

    def container_named(self, container: str) -> Container | None:
        found = [described for described in self.containers if described.name == container]
        assert len(found) <= 1, f"{self.key} declares {len(found)} containers named {container!r}"
        return found[0] if found else None


class Manifest(FrozenStrictBaseModel):
    objects: list[ManifestObject]

    @classmethod
    def parse(cls, rendered: str) -> Manifest:
        return cls(objects=[document for document in yaml.safe_load_all(rendered) if document])

    @property
    def by_key(self) -> dict[str, ManifestObject]:
        return {described.key: described for described in self.objects}

    def flag_value(self, flag: str, *, stateful_set: str, container: str) -> str | None:
        described = self.by_key.get(f"StatefulSet/{stateful_set}")
        if described is None:
            return None

        found = described.container_named(container)
        if found is None or flag not in found.command:
            return None

        value_index = found.command.index(flag) + 1
        assert value_index < len(found.command), (
            f"container {container!r} of {stateful_set} ends its command with {flag}, which takes a value, so this "
            f"launch cannot tell what the installed release was told"
        )
        return found.command[value_index]

    def state_file(self, *, stateful_set: str, container: str) -> Path | None:
        named = self.flag_value(STATE_FILE_FLAG, stateful_set=stateful_set, container=container)
        return Path(named) if named is not None else None

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.train.group import TrainerController
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


def make_inference_controller(
    args: Any,
    *,
    engine_provider: BaseWorkerProvider | None = None,
    router_providers: Sequence[BaseWorkerProvider] = (),
    **overrides: Any,
) -> InferenceController:
    controller = InferenceController(
        args,
        engine_provider=engine_provider if engine_provider is not None else MagicMock(spec=BaseWorkerProvider),
        router_providers=router_providers,
    )
    _apply_overrides(controller, overrides=overrides)
    return controller


def make_trainer_controller(
    *,
    args: Any = None,
    launch_args: Any = None,
    role: str = "actor",
    with_ref: bool = False,
    cell_provider: BaseWorkerProvider | None = None,
    cell_operations: BaseCellOperations | None = None,
    **overrides: Any,
) -> TrainerController:
    controller = TrainerController(
        launch_args if launch_args is not None else Namespace(),
        cell_provider=cell_provider if cell_provider is not None else MagicMock(spec=BaseWorkerProvider),
        cell_operations=cell_operations if cell_operations is not None else MagicMock(spec=BaseCellOperations),
        role=role,
        with_ref=with_ref,
    )
    if args is not None:
        controller.args = args
    _apply_overrides(controller, overrides=overrides)
    return controller


def _apply_overrides(controller: object, *, overrides: dict[str, Any]) -> None:
    for name, value in overrides.items():
        setattr(controller, name, value)

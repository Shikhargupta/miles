from __future__ import annotations

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.specs.trainer_identity import compute_trainer_roles
from miles.utils.workers.types import DeploymentIdentity
from miles.utils.workers.worker_provider.simple import parse_host_and_port
from miles.utils.workers.worker_spec import HostAndPort

TRAINER_CONTROLLER_ADDRS_FLAG = "--trainer-controller-addrs"
INFERENCE_CONTROLLER_ADDRS_FLAG = "--inference-controller-addrs"
INFERENCE_ROUTER_ADDRS_FLAG = "--inference-router-addrs"


def trainer_controller_urls(args, *, role: str) -> list[str] | None:
    if (entries := args.trainer_controller_addrs) is None:
        return None
    urls = _group_by_key(entries, keys=compute_trainer_roles(args), flag=TRAINER_CONTROLLER_ADDRS_FLAG).get(role, [])
    assert len(urls) == 1, (
        f"{TRAINER_CONTROLLER_ADDRS_FLAG} names {len(urls)} controllers for trainer {role!r}, but every trainer "
        f"of a run is driven through exactly one of them"
    )
    return urls


def inference_controller_urls(args) -> list[str] | None:
    if (entries := args.inference_controller_addrs) is None:
        return None
    assert len(entries) == 1, (
        f"{INFERENCE_CONTROLLER_ADDRS_FLAG} names {len(entries)} controllers, but a run drives exactly one of them, "
        f"and every other engine deployment registers its cells into that one"
    )
    return list(entries)


def static_router_addrs(args) -> dict[str, HostAndPort] | None:
    if (entries := args.inference_router_addrs) is None:
        return None

    model_names = [model.name for model in resolve_sglang_config(args).models]
    urls_by_model = _group_by_key(entries, keys=model_names, flag=INFERENCE_ROUTER_ADDRS_FLAG)
    missing = [name for name in model_names if not urls_by_model.get(name)]
    assert not missing, (
        f"every model is served by its own router, so {INFERENCE_ROUTER_ADDRS_FLAG} needs an entry for {missing} "
        f"(the models come from the resolved sglang config: {model_names})"
    )
    return {name: parse_host_and_port(_one(urls_by_model[name], name=name)) for name in model_names}


def assert_deployment_names_this_run(identity: DeploymentIdentity, *, args, flag: str) -> None:
    assert identity.run_uuid == args.run_uuid, (
        f"{flag} names the {identity.deploy_component} deployment of run {identity.run_uuid}, but this launch "
        f"drives run {args.run_uuid}: every deployment a split run reaches has to be a deployment of that same run, "
        f"or its weight updates and its rollout samples belong to different runs"
    )


def assert_routers_belong_to_inference_deployment(identity: DeploymentIdentity, *, args) -> None:
    given = {name: f"{addr.host}:{addr.port}" for name, addr in (static_router_addrs(args) or {}).items()}
    assert given == identity.router_addrs, (
        f"{INFERENCE_ROUTER_ADDRS_FLAG} names {given}, but the deployment reached through "
        f"{INFERENCE_CONTROLLER_ADDRS_FLAG} serves {identity.router_addrs}: the routers live with the engines they "
        f"serve, so weights would be broadcast to one inference deployment while rollout samples come from another"
    )


def _one(urls: list[str], *, name: str) -> str:
    assert len(urls) == 1, f"{name} is served by one router, but {len(urls)} addresses name it: {urls}"
    return urls[0]


def _group_by_key(entries: list[str], *, keys: list[str], flag: str) -> dict[str, list[str]]:
    prefixed = [split for entry in entries if (split := _split_key(entry)) is not None]
    assert len(prefixed) in (
        0,
        len(entries),
    ), f"{flag} must be uniformly bare addresses or uniformly '<name>=<address>' entries (got {entries})"

    if not prefixed:
        assert len(entries) == 1, f"{flag} takes bare addresses only when one entry is expected (got {entries})"
        return {keys[0]: list(entries)}

    grouped: dict[str, list[str]] = {}
    for key, addr in prefixed:
        assert key in keys, f"{flag} names {key!r}, which is not one of {keys}"
        grouped.setdefault(key, []).append(addr)
    return grouped


def _split_key(entry: str) -> tuple[str, str] | None:
    prefix, separator, rest = entry.partition("=")
    if not separator or "://" in prefix or ":" in prefix or "/" in prefix:
        return None
    return prefix, rest

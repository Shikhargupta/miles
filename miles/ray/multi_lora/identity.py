"""Registration-scoped identities for the Multi-LoRA operation backend."""

import uuid

RID_SEPARATOR = "::"


def make_rid(adapter_name: str, registration_id: str) -> str:
    """Mint a request ID inside one exact adapter registration."""
    return f"{adapter_name}{RID_SEPARATOR}{registration_id}{RID_SEPARATOR}{uuid.uuid4().hex}"


def rid_prefix(adapter_name: str, registration_id: str) -> str:
    """Return the abort namespace for one exact adapter registration."""
    return f"{adapter_name}{RID_SEPARATOR}{registration_id}{RID_SEPARATOR}"


def parse_adapter(rid: str) -> str:
    """Extract the adapter name from a registration-scoped request ID."""
    return rid.split(RID_SEPARATOR, 1)[0]


def serving_lora_name(adapter_name: str, registration_id: str) -> str:
    """Return the engine-side name for one exact adapter registration."""
    return f"__miles_adapter_{adapter_name}_{registration_id}"


def cache_extra_key(adapter_name: str, registration_id: str, serving_version: int) -> str:
    """Return the registration- and version-scoped KV-cache namespace."""
    return f"{adapter_name}:{registration_id}:v{serving_version}"

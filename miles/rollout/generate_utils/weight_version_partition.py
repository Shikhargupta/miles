# Namespacing generation requests by the weight version seen when a session
# started keeps radix-cache prefix reuse within one weight generation: pause
# modes that skip engine-side cache flushes (in_place) would otherwise let KV
# computed under old weights serve unrelated new requests indefinitely.

WEIGHT_VERSION_EXTRA_KEY_METADATA_KEY: str = "weight_version_extra_key"


def format_weight_version_extra_key(weight_version: int | None) -> str:
    return f"weight-version:{0 if weight_version is None else weight_version}"


def observe_weight_version(current: int | None, meta_info: dict) -> int | None:
    raw = meta_info.get("weight_version")
    try:
        version = int(raw)
    except (TypeError, ValueError):
        return current
    return version if current is None else max(current, version)

# Environment variable helpers used across scripts.

import os


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise ValueError(f"Missing required env var: {name}")
    return value


def get_int(name: str, default: int | None = None) -> int:
    value = get_env(name, None if default is None else str(default))
    if value is None:
        raise ValueError(f"Missing required env var: {name}")
    return int(value)


def get_float(name: str, default: float | None = None) -> float:
    value = get_env(name, None if default is None else str(default))
    if value is None:
        raise ValueError(f"Missing required env var: {name}")
    return float(value)


def get_bool(name: str, default: bool | None = None) -> bool:
    if default is None:
        value = require_env(name)
    else:
        value = get_env(name, str(default))
    return value.lower() in {"1", "true", "yes", "y"}

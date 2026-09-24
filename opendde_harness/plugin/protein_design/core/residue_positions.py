"""Explicit conversions at user/LLM boundaries; compute contracts stay zero-based."""

from collections.abc import Mapping
from typing import Any


def external_positions(value: Any) -> Any:
    """Convert only a position list or chain/region map, never a whole payload."""
    if isinstance(value, Mapping):
        return {key: external_positions(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [external_positions(item) for item in value]
    return int(value) + 1


def internal_positions(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: internal_positions(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [internal_positions(item) for item in value]
    position = int(value)
    if position < 1:
        raise ValueError("Residue positions must be one-based (at least 1)")
    return position - 1


def external_mutations(values: list) -> list:
    return [
        {**item, "position": int(item["position"]) + 1 if item.get("position") is not None else None}
        if isinstance(item, Mapping)
        else [item[0], int(item[1]) + 1, *item[2:]]
        for item in values
    ]

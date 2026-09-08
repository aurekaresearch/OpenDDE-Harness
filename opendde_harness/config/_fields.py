"""Pydantic field introspection and the JSON write path shared by the ``update_*`` modules."""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from opendde_harness.utils.atomic_io import atomic_replace


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Rewrite ``path`` with ``data`` as indented UTF-8 JSON via temp file + rename."""
    atomic_replace(path, json.dumps(data, indent=2, ensure_ascii=False))


def unwrap_optional(annotation: Any) -> Any:
    """Strip ``Optional[X]`` / ``X | None`` down to ``X``."""
    import types as _types

    origin = typing.get_origin(annotation)
    if origin is Union or origin is getattr(_types, "UnionType", None):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def is_model_class(ann: Any) -> bool:
    return isinstance(ann, type) and issubclass(ann, BaseModel)


def annotation_str(ann: Any) -> str:
    """Compact type string for the ``show`` command."""
    ann = unwrap_optional(ann)
    origin = typing.get_origin(ann)
    if origin is typing.Literal:
        return "Literal"
    if origin is list:
        args = typing.get_args(ann)
        return f"list[{annotation_str(args[0])}]" if args else "list"
    if origin is dict:
        args = typing.get_args(ann)
        if args and len(args) == 2:
            return f"dict[{annotation_str(args[0])}, {annotation_str(args[1])}]"
        return "dict"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann)


_SECRET_EXACT = {"token", "secret", "password", "api_key"}
_SECRET_SUFFIXES = ("_token", "_secret", "_key", "_password")

# Names that should be redacted but neither match _SECRET_EXACT nor end in a
# secret suffix. Today this only covers Gemini's ``api_key_list`` (suffix is
# ``_list``, not ``_key``). Delete entries here as schema.py grows the
# ``json_schema_extra={"secret": True}`` marker on the underlying fields.
_KNOWN_SECRET_FIELDS: set[str] = {"api_key_list"}


def is_secret_field(field_name: str, field_info: Any) -> bool:
    """Detect secret fields, in priority order:

    1. Explicit: ``field_info.json_schema_extra.get('secret') is True``
    2. Patch set: ``_KNOWN_SECRET_FIELDS`` (workaround for fields the
       suffix heuristic misses, e.g. Gemini's ``api_key_list``).
    3. Exact match (``token`` / ``secret`` / ``password`` / ``api_key``).
    4. Suffix match (``_token`` / ``_secret`` / ``_key`` / ``_password``).
    """
    extra = getattr(field_info, "json_schema_extra", None)
    if isinstance(extra, dict) and extra.get("secret") is True:
        return True
    if field_name in _KNOWN_SECRET_FIELDS:
        return True
    if field_name in _SECRET_EXACT:
        return True
    return any(field_name.endswith(suf) for suf in _SECRET_SUFFIXES)


def coerce_value(value: Any, annotation: Any) -> Any:
    """Pre-Pydantic coercion for CLI string inputs.

    Handles bool /
    int / float / list / dict surfaces so the same ``--flag value`` UX works
    for both groups.
    """
    if not isinstance(value, str):
        return value

    base = unwrap_optional(annotation)

    if base is bool:
        v = value.strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False
        return value

    if base is int:
        try:
            return int(value)
        except ValueError:
            return value

    if base is float:
        try:
            return float(value)
        except ValueError:
            return value

    origin = typing.get_origin(base)
    if origin is list:
        v = value.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]

    if origin is dict:
        v = value.strip()
        if v.startswith("{") and v.endswith("}"):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return value

    return value


def field_default(field_info: Any) -> Any:
    """Resolve a Pydantic FieldInfo's effective default (call factory if any)."""
    if field_info.default_factory is not None:
        try:
            return field_info.default_factory()
        except Exception:
            return None
    if field_info.default is PydanticUndefined:
        return None
    return field_info.default


def flatten_fields(cls: type[BaseModel], prefix: str = "") -> dict[str, dict[str, Any]]:
    """Flatten a model's fields to ``path -> spec``, recursing into nested models."""
    out: dict[str, dict[str, Any]] = {}
    for fname, finfo in cls.model_fields.items():
        ann = unwrap_optional(finfo.annotation)
        path = f"{prefix}{fname}"
        if is_model_class(ann):
            out.update(flatten_fields(ann, prefix=f"{path}."))
            continue
        description = finfo.description or ""
        origin = typing.get_origin(ann)
        if origin is typing.Literal and not description:
            choices = ", ".join(str(a) for a in typing.get_args(ann))
            description = f"Choices: {choices}"
        out[path] = {
            "type": annotation_str(ann),
            "default": field_default(finfo),
            "is_secret": is_secret_field(fname, finfo),
            "description": description,
        }
    return out


def flatten_instance(instance: BaseModel, prefix: str = "") -> dict[str, Any]:
    """Flatten a Pydantic instance to ``path -> value``."""
    out: dict[str, Any] = {}
    for fname in type(instance).model_fields:
        val = getattr(instance, fname)
        path = f"{prefix}{fname}"
        if isinstance(val, BaseModel):
            out.update(flatten_instance(val, prefix=f"{path}."))
        else:
            out[path] = val
    return out


def walk_nested_path(model_cls: type[BaseModel], dotted_key: str) -> tuple[type[BaseModel], str]:
    """Walk ``a.b.c`` through nested ``BaseModel`` classes."""
    segs = dotted_key.split(".")
    cls: type[BaseModel] = model_cls
    for seg in segs[:-1]:
        finfo = cls.model_fields.get(seg)
        if finfo is None:
            raise KeyError(f"Unknown nested field '{seg}' in {cls.__name__}")
        ann = unwrap_optional(finfo.annotation)
        if not is_model_class(ann):
            raise KeyError(f"Field '{seg}' in {cls.__name__} is not a nested model")
        cls = ann
    leaf = segs[-1]
    if leaf not in cls.model_fields:
        raise KeyError(f"Unknown field '{leaf}' in {cls.__name__}")
    return cls, leaf


def set_nested(dotted_key: str, value: Any, target: dict[str, Any]) -> Any:
    """Set ``target[a][b][...][leaf] = value``; return previous value."""
    segs = dotted_key.split(".")
    cursor = target
    for seg in segs[:-1]:
        nxt = cursor.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[seg] = nxt
        cursor = nxt
    prev = cursor.get(segs[-1])
    cursor[segs[-1]] = value
    return prev

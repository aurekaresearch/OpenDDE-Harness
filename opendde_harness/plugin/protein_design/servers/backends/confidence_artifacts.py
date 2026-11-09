"""Lossless compact storage for OpenDDE ``full_data`` confidence artifacts.

OpenDDE writes ``*_full_data_sample_*.json`` after ``np.round(x, 2)`` on float32
tensors (``runner/dumper.py``), so a PAE of 20.38 is written as the float64 text of
its float32 value, ``20.3799991607666``. The values carry only 1/100 resolution
but cost ~17 bytes each, and these files dominate artifact disk use. This module
re-encodes them as compressed ``.npz`` where every float array that holds whole
hundredths is stored as an integer count of hundredths (int16/int32) plus the
float width the hundredths were computed in. Decoding divides by 100 in that same
width, which yields exactly the float that ``json.loads`` returns for the original
text, so readers see identical values.

Anything that does not fit that scheme is kept losslessly: float arrays that
round-trip through float32 are stored as float32, other floats as float64, and
values that are not rectangular numeric arrays are stored as their JSON text.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FULL_DATA_FORMATS = frozenset({"json", "npz"})
NPZ_FORMAT_TAG = "opendde-harness-full-data-v1"

_FORMAT_KEY = "__format__"
_SCALE_PREFIX = "__scale__/"
_JSON_PREFIX = "__json__/"
_HUNDREDTHS = 100


def load_first_json_object(path: str | Path) -> Any:
    """Load the first complete JSON value and tolerate stale trailing bytes.

    Like the loss scorer's reader, this also accepts integral floats written as
    ``6.`` (valid NumPy text, invalid JSON) and fails on anything else.
    """
    text = Path(path).read_text(encoding="utf-8").lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        normalized = re.sub(r"(?<![\w.])(-?\d+)\.(?=\s*[,}\]])", r"\1.0", text)
        if normalized == text:
            raise
        value, _ = json.JSONDecoder().raw_decode(normalized)
    return value


def is_compact_full_data(path: str | Path) -> bool:
    """Return True when ``path`` is an npz written by :func:`compact_full_data`."""
    if not str(path).endswith(".npz"):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return _FORMAT_KEY in data.files and str(data[_FORMAT_KEY]) == NPZ_FORMAT_TAG
    except (OSError, ValueError):
        return False


def load_full_data(path: str | Path) -> dict[str, Any]:
    """Load a full_data artifact written as OpenDDE JSON or compact npz.

    JSON returns the parsed object unchanged. Compact npz returns a dict with the
    same keys, where arrays come back as numpy arrays whose values equal the
    original JSON numbers exactly.
    """
    if not str(path).endswith(".npz"):
        data = load_first_json_object(path)
        if not isinstance(data, dict):
            raise ValueError(f"full_data payload is not a JSON object: {path}")
        return data

    with np.load(path, allow_pickle=False) as archive:
        if _FORMAT_KEY not in archive.files or str(archive[_FORMAT_KEY]) != NPZ_FORMAT_TAG:
            raise ValueError(f"Not a compact full_data artifact: {path}")
        scales = {name[len(_SCALE_PREFIX) :]: archive[name] for name in archive.files if name.startswith(_SCALE_PREFIX)}
        result: dict[str, Any] = {}
        for name in archive.files:
            if name == _FORMAT_KEY or name.startswith(_SCALE_PREFIX):
                continue
            if name.startswith(_JSON_PREFIX):
                result[name[len(_JSON_PREFIX) :]] = json.loads(str(archive[name]))
                continue
            value = archive[name]
            scale = scales.get(name)
            if scale is not None:
                value = _decode_hundredths(value, scale)
            result[name] = value
        return result


def _decode_hundredths(counts: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (counts.astype(scale.dtype) / scale).astype(np.float64)


def _smallest_int_dtype(values: np.ndarray) -> np.dtype:
    if values.size == 0:
        return np.dtype(np.int32)
    low, high = int(values.min()), int(values.max())
    for dtype in (np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if info.min <= low and high <= info.max:
            return np.dtype(dtype)
    raise OverflowError("integer values exceed int64")


def _encode_value(key: str, value: Any) -> dict[str, np.ndarray]:
    """Encode one JSON value as one or two npz members without losing information."""
    if isinstance(value, (str, dict)) or value is None:
        return {_JSON_PREFIX + key: np.array(json.dumps(value))}
    try:
        array = np.asarray(value)
    except ValueError:  # ragged nested lists
        return {_JSON_PREFIX + key: np.array(json.dumps(value))}

    if array.dtype == np.bool_:
        return {key: array}
    if array.dtype.kind in "iu":
        return {key: array.astype(_smallest_int_dtype(array))}
    if array.dtype.kind != "f":
        return {_JSON_PREFIX + key: np.array(json.dumps(value))}

    array = array.astype(np.float64)
    if np.all(np.isfinite(array)):
        scaled = np.rint(array * _HUNDREDTHS)
        # 2**24 keeps every count exact in float32 as well as float64.
        if np.abs(scaled).max(initial=0.0) < 2**24:
            ints = scaled.astype(np.int64)
            # float32 first (what OpenDDE writes), then plain float64 hundredths.
            # Keep the fixed-point form only if decoding reproduces every float exactly.
            for width in (np.float32, np.float64):
                scale = np.array(_HUNDREDTHS, dtype=width)
                if np.array_equal(_decode_hundredths(ints, scale), array):
                    return {key: ints.astype(_smallest_int_dtype(ints)), _SCALE_PREFIX + key: scale}
    as_f32 = array.astype(np.float32)
    if np.array_equal(as_f32.astype(np.float64), array, equal_nan=True):
        return {key: as_f32}
    return {key: array}


def _values_equal(original: Any, decoded: Any) -> bool:
    if isinstance(original, (str, dict)) or original is None or not isinstance(decoded, np.ndarray):
        return original == decoded
    expected = np.asarray(original)
    if expected.dtype.kind == "f" or decoded.dtype.kind == "f":
        return expected.shape == decoded.shape and np.array_equal(
            expected.astype(np.float64), decoded.astype(np.float64), equal_nan=True
        )
    return expected.shape == decoded.shape and np.array_equal(expected, decoded)


def encode_full_data(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the npz members for a full_data mapping."""
    if _FORMAT_KEY in data or any(k.startswith((_SCALE_PREFIX, _JSON_PREFIX)) for k in data):
        raise ValueError("full_data uses a reserved key name")
    members: dict[str, np.ndarray] = {_FORMAT_KEY: np.array(NPZ_FORMAT_TAG)}
    for key, value in data.items():
        members.update(_encode_value(key, value))
    return members


def compact_full_data(json_path: str | Path, *, remove_json: bool = True) -> Path:
    """Re-encode one full_data JSON as lossless compressed npz.

    The npz is written to a temporary name, read back, and compared with the
    source JSON value by value before it replaces anything. On any failure the
    original JSON is left in place and its path is returned, so callers never
    lose confidence data because of this step.
    """
    source = Path(json_path)
    target = source.with_suffix(".npz")
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    try:
        data = load_first_json_object(source)
        if not isinstance(data, dict):
            raise ValueError("full_data payload is not a JSON object")
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **encode_full_data(data))
        decoded = load_full_data(tmp)
        if decoded.keys() != data.keys() or not all(_values_equal(data[k], decoded[k]) for k in data):
            raise ValueError("decoded npz does not match the source JSON")
        os.replace(tmp, target)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        logger.warning("Keeping %s as JSON; compact npz encoding failed: %s", source, exc)
        return source
    if remove_json:
        source.unlink(missing_ok=True)
    return target


def compact_full_data_dir(root: str | Path, *, remove_json: bool = True) -> dict[Path, Path]:
    """Compact every ``*full_data_sample_*.json`` below ``root``.

    Returns a mapping from each original JSON path to the path that now holds its
    data (the npz on success, the untouched JSON otherwise).
    """
    mapping: dict[Path, Path] = {}
    for json_path in sorted(Path(root).glob("**/*full_data_sample_*.json")):
        mapping[json_path] = compact_full_data(json_path, remove_json=remove_json)
    return mapping

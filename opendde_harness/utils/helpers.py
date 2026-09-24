"""Utility functions for opendde_harness."""

import hashlib
import json
import math
import re
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from loguru import logger

from opendde_harness.providers import messages as msg

# Workspace sync runs before the CLI decides logger.enable/disable("opendde_harness"),
# so an unscoped debug in this module would spam stderr through loguru's
# default sink on every first run. A later logger.enable("opendde_harness") still
# lifts this rule (loguru drops descendant rules whenever a parent rule is
# set), but the CLI flips logging only after its startup sync -- so the
# per-file detail below reaches callers that enable logging before syncing
# (tests, embedders), not the first sync of a `--logs` run.
logger.disable(__name__)


def detect_image_mime(data: bytes) -> str | None:
    """Detect image MIME type from magic bytes, ignoring file extension."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


# Vision models bill images by patch area, not by the size of the transport
# encoding. Counting a data URI as text charges ~350x the real cost (a 1000x1000
# JPEG is ~1.3k image tokens but ~460k base64 characters), which starves the
# history budget and can trip emergency shrinking on a prompt that would have
# fit comfortably.
_IMAGE_PATCH_PX = 28
_IMAGE_TOKEN_CAP = 1568


def _image_pixel_size(data: bytes) -> tuple[int, int] | None:
    """Pixel dimensions from an image header, or None if not derivable.

    Header-only parsing on purpose: the caller has a whole image in memory
    already and this runs on every budget probe, so decoding pixels (or pulling
    in an imaging library) would cost far more than the estimate is worth.
    WebP is deliberately absent -- its three chunk variants need more parsing
    than the fallback is worth.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data[:3] == b"\xff\xd8\xff":
        # Walk JPEG segments to the first frame header; SOF carries the size.
        i = 2
        while i + 1 < len(data):
            if data[i] != 0xFF:
                return None
            marker = data[i + 1]
            # 0xFF is a fill byte, legal in any run before a marker.
            if marker == 0xFF:
                i += 1
                continue
            # Standalone markers carry no length field, so the generic
            # "skip the segment" step below would read their *payload* as a
            # length and desync the walk. TEM (0x01) and RST0-7 (0xD0-0xD7)
            # are the ones that can precede SOF.
            if marker == 0x01 or 0xD0 <= marker <= 0xD9:
                i += 2
                continue
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if i + 9 > len(data):
                    return None
                return (
                    int.from_bytes(data[i + 7 : i + 9], "big"),
                    int.from_bytes(data[i + 5 : i + 7], "big"),
                )
            if i + 4 > len(data):
                return None
            length = int.from_bytes(data[i + 2 : i + 4], "big")
            if length < 2:
                return None  # malformed: a segment length includes its own 2 bytes
            i += 2 + length
    return None


def estimate_image_tokens(width: int, height: int, cap: int = _IMAGE_TOKEN_CAP) -> int:
    """Image tokens for a ``width`` x ``height`` image, same order of magnitude
    across vendors and biased high.

    Anthropic's own formula (28x28 patches, capped at 1568 for the standard
    tier). Exact for Claude; ~10% high for OpenAI's 512px tiles; ~2.25x high for
    Doubao 2.x, which moved to 42x42 patches. Over-estimating is the safe
    direction for a budget guard -- under-estimating overflows the context.
    """
    if width <= 0 or height <= 0:
        return cap
    patches = math.ceil(width / _IMAGE_PATCH_PX) * math.ceil(height / _IMAGE_PATCH_PX)
    return min(patches, cap)


def estimate_content_part_tokens(part: Any) -> int | None:
    """Token estimate for an image content block, or None for anything else.

    A block whose header does not parse is charged the ceiling: over-estimating
    is the safe direction for a budget guard, and under-estimating overflows the
    context.
    """
    if not msg.is_image(part):
        return None
    head = msg.image_payload(part)
    if head is None:
        return _IMAGE_TOKEN_CAP
    size = _image_pixel_size(head)
    return estimate_image_tokens(*size) if size else _IMAGE_TOKEN_CAP


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def safe_filename(name: str) -> str:
    """Replace unsafe path characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip()


def split_message(content: str, max_len: int = 2000) -> list[str]:
    """
    Split content into chunks within max_len, preferring line breaks.

    Args:
        content: The text content to split.
        max_len: Maximum length per chunk (default 2000 for Discord compatibility).

    Returns:
        List of message chunks, each within max_len.
    """
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # Try to break at newline first, then space, then hard break
        pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


# Token counts keyed by a digest of the text they were counted on. A session's
# messages are re-estimated on every turn -- the manifest, the budget, and each
# iteration of the trim loop all ask again -- and encoding is the cost: one
# pass over a 100k-token history took a third of a second on the event loop,
# and the trim loop took that pass once per dropped exchange. Hashing the text
# is two orders of magnitude cheaper than encoding it, and a message that has
# not changed encodes to the same count. Keyed by content rather than by
# object so an edit in place (a tool body elided by the loop) is a miss, not a
# stale hit. Bounded, so a long session cannot grow it without limit.
_TOKEN_COUNT_CACHE: OrderedDict[bytes, int] = OrderedDict()
_TOKEN_COUNT_CACHE_MAX = 16384
# History selection counts in a worker thread while the loop counts on its own; a
# lookup racing an eviction raised KeyError out of move_to_end.
_TOKEN_COUNT_LOCK = threading.Lock()


def count_text_tokens(payload: str) -> int:
    """tiktoken's count for ``payload``, cached by content."""
    if not payload:
        return 0
    key = hashlib.blake2b(payload.encode("utf-8", "surrogatepass"), digest_size=16).digest()
    with _TOKEN_COUNT_LOCK:
        cached = _TOKEN_COUNT_CACHE.get(key)
        if cached is not None:
            _TOKEN_COUNT_CACHE.move_to_end(key)
            return cached
    try:
        count = len(_encode(payload))
    except Exception:
        count = len(payload) // 4
    with _TOKEN_COUNT_LOCK:
        _TOKEN_COUNT_CACHE[key] = count
        if len(_TOKEN_COUNT_CACHE) > _TOKEN_COUNT_CACHE_MAX:
            _TOKEN_COUNT_CACHE.popitem(last=False)
    return count


def tokenizer_is_loaded() -> bool:
    """Whether counting is a local operation in this process, right now.

    tiktoken fetches its vocabulary over the network the first time it is asked
    for one, and a fresh install has no cache to read. Inside a turn that costs
    nothing anybody notices -- the turn is a network call already -- but a path
    that must answer offline cannot reach for the internet to do it: measured
    with an empty cache, the first count opened a connection and only then fell
    back to characters over four.

    Callers that must not fetch ask this first and use the character rule
    otherwise, which is the same answer ``count_text_tokens`` gives when the
    tokenizer is away. False on any version of tiktoken that keeps its built
    encodings somewhere else, which costs an estimate its precision and never
    its correctness.
    """
    try:
        return "cl100k_base" in tiktoken.registry.ENCODINGS
    except Exception:
        return False


def _encode(payload: str) -> list[int]:
    """``payload`` as tokens, with every special-token spelling read as text.

    tiktoken refuses ``<|endoftext|>`` in user text by default, and the refusal
    is an exception, not a count: every caller fell through to its own rough
    fallback the moment a user pasted those eleven characters. Nothing here
    ever sends a control token, so the spelling is just text.
    """
    return tiktoken.get_encoding("cl100k_base").encode(payload, disallowed_special=())


def take_tokens(payload: str, limit: int) -> tuple[str, int]:
    """The longest prefix of ``payload`` within ``limit`` tokens, and its cost.

    A prefix of the source, always. Cutting at ``limit * 4`` characters is not
    a token limit -- dense text, CJK above all, stays several times over the
    budget it was supposedly cut to -- and cutting at a token boundary is not a
    character boundary: decoding across one puts a U+FFFD where the user's last
    character was, which is text they never wrote, stored in their session.
    The incomplete bytes are dropped instead, and the result is recounted,
    because a cut string need not re-encode to what its tokens cost inside the
    whole.

    The cost comes back with the text because a caller spending one shared
    allowance cannot work it out afterwards: counting the returned prefix asks
    the same estimator that answers characters-over-four when the tokenizer is
    away, which is not what the cut was measured in. Fifteen parts that each
    "fit" that way totalled 67,000 tokens of a 20,000 budget.
    """
    if limit <= 0 or not payload:
        return "", 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = _encode(payload)
        if len(tokens) <= limit:
            return payload, len(tokens)
        text = _utf8_prefix(encoding.decode_bytes(tokens[:limit]))
        if text:
            exact = len(_encode(text))
            if exact <= limit:
                return text, exact
    except Exception:
        pass
    # One byte is at most one token, so a byte budget is a token budget nothing
    # can exceed -- the conservative answer when the tokenizer is unavailable,
    # or when its slice re-encodes over the allowance. Charged in bytes for the
    # same reason it was cut in bytes.
    text = _utf8_prefix(payload.encode("utf-8")[:limit])
    return text, len(text.encode("utf-8"))


def truncate_to_tokens(payload: str, limit: int) -> str:
    """The longest prefix of ``payload`` that costs at most ``limit`` tokens."""
    return take_tokens(payload, limit)[0]


def _utf8_prefix(raw: bytes) -> str:
    """``raw`` decoded, less any character its last bytes only start.

    The bytes are a prefix of a valid UTF-8 string, so the only sequence that
    can be incomplete is the final one; dropping it leaves a prefix of the
    source rather than a replacement character.
    """
    return raw.decode("utf-8", "ignore")


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate prompt tokens with tiktoken.

    Additive over messages, so each message's count comes from the cache and a
    prompt that grew by one message costs one encode. Counting the joined text
    in one pass gave a marginally different total at each message boundary; an
    estimate does not owe that precision, and the per-turn cost it carried was
    the TUI stutter.
    """
    total = sum(estimate_message_tokens(msg) for msg in messages)
    if tools:
        total += count_text_tokens(json.dumps(tools, ensure_ascii=False))
    return total


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate prompt tokens contributed by one message.

    Text counts as text; an image by its patch area; a tool call by its name
    and its arguments, which is what the request spells out. Replayed thinking
    counts as the text the model wrote, once -- never its signature, an opaque
    blob many times the size of the reasoning it encodes and not billed as
    prompt text: measured on the Codex login, counting it made a 31k-token
    prompt read as 161k, and the fitter that trusted the number elided the
    history.
    """
    content = message.get("content")
    parts: list[str] = []
    extra_tokens = 0
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == msg.TEXT:
                parts.append(str(block.get("text") or ""))
            elif kind == msg.THINKING:
                parts.append(str(block.get("thinking") or ""))
            elif kind == msg.TOOL_CALL:
                parts.append(str(block.get("name") or ""))
                parts.append(json.dumps(block.get("arguments") or {}, ensure_ascii=False))
            elif (image_tokens := estimate_content_part_tokens(block)) is not None:
                extra_tokens += image_tokens
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))

    # The pairing a tool result carries, which the request spells out beside
    # its body.
    for key in ("toolName", "toolCallId"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)

    payload = "\n".join(part for part in parts if part)
    if not payload:
        return max(1, extra_tokens)
    return max(1, count_text_tokens(payload) + extra_tokens)


def estimate_prompt_tokens_chain(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    """Estimate prompt tokens via provider counter first, then tiktoken fallback."""
    provider_counter = getattr(provider, "estimate_prompt_tokens", None)
    if callable(provider_counter):
        try:
            tokens, source = provider_counter(messages, tools, model)
            if isinstance(tokens, (int, float)) and tokens > 0:
                return int(tokens), str(source or "provider_counter")
        except Exception:
            pass

    estimated = estimate_prompt_tokens(messages, tools)
    if estimated > 0:
        return int(estimated), "tiktoken"
    return 0, "none"


def sync_workspace_templates(
    workspace: Path,
    silent: bool = False,
    *,
    assistant_dir: Path | None = None,
    skills_dir: Path | None = None,
) -> list[str]:
    """Initialize instance assistant templates; never migrate workspace files."""
    from importlib.resources import files as pkg_files

    from opendde_harness.config.paths import assert_storage_ready, get_workspace_storage

    storage = get_workspace_storage(workspace)
    assert_storage_ready(storage)
    assistant = assistant_dir if assistant_dir is not None else storage.assistant
    skills = skills_dir if skills_dir is not None else storage.skills

    try:
        tpl = pkg_files("opendde_harness") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []
    existed = 0

    def _write(src, dest: Path):
        nonlocal existed
        if dest.exists():
            existed += 1
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8") if src else "", encoding="utf-8")
        added.append(str(dest))

    _write(tpl / "SOUL.md", assistant / "soul.md")
    _write(tpl / "AGENTS.md", assistant / "agent.md")
    _write(tpl / "TOOLS.md", assistant / "TOOLS.md")
    skills.mkdir(parents=True, exist_ok=True)

    if added:
        for name in added:
            logger.debug("workspace sync: created {}", name)
    if added and not silent:
        from rich.console import Console

        _c = Console(stderr=True)
        label = "Initialized assistant" if existed == 0 else "Updated assistant templates"
        _c.print(f"  [dim]{label} ({len(added)} file{'s' if len(added) != 1 else ''})[/dim]")
    return added

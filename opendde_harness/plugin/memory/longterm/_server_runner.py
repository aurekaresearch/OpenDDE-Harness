"""Start the long-term memory server with OpenDDE Harness's configurable OpenAI API protocol adapter."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Literal

from opendde_harness.plugin.memory.longterm._library import CONFIG_FILENAME, patch_llm_client, start_server


def configured_api_mode(root: Path) -> Literal["responses", "chat"]:
    """Read ``[llm].api_mode``; Responses is the intentional default."""
    with (root / CONFIG_FILENAME).open("rb") as handle:
        value = (tomllib.load(handle).get("llm") or {}).get("api_mode", "responses")
    if value not in {"responses", "chat"}:
        raise ValueError(
            f"invalid [llm].api_mode={value!r}; expected 'responses' or 'chat'"
        )
    return value


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        root = Path(args[args.index("--root") + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("--root PATH is required") from exc

    mode = configured_api_mode(root)
    if mode == "responses":
        from ._responses_llm import ResponsesLLMClient

        patch_llm_client(ResponsesLLMClient)

    start_server(root)


if __name__ == "__main__":
    main()

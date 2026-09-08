"""Target-only MSA search through Protenix's online MMseqs service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from opendde_harness.plugin.protein_design.core.constants import CANONICAL_AMINO_ACIDS
from opendde_harness.plugin.protein_design.core.external import (
    DEFAULT_MSA_SERVER_URL,
    MSA_SERVICE,
    ExternalServiceUnavailableError,
    connection_reason,
)


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return normalized[:80] or "target"


def _normalize_sequence(value: str) -> str:
    sequence = "".join(value.split()).upper()
    invalid = sorted(set(sequence) - CANONICAL_AMINO_ACIDS)
    if not sequence:
        raise ValueError("target MSA search requires a non-empty target sequence")
    if invalid:
        raise ValueError(
            "target MSA search accepts only canonical amino acids; invalid: "
            + ", ".join(invalid)
        )
    return sequence


def _alignment_depth(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith(">"))


def _validate_a3m_query(path: Path, sequence: str) -> None:
    query: list[str] = []
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
        elif started and not line.startswith("#"):
            query.append(line.strip())
    if "".join(query) != sequence:
        raise RuntimeError("MSA query sequence does not match the requested target")


def _normalize_protenix_result(output_dir: Path, result_dir: Path, sequence: str) -> None:
    unpaired = result_dir / "non_pairing.a3m"
    if unpaired.is_file() and _alignment_depth(unpaired) > 1:
        return
    # Protenix returns a per-query A3M rather than ColabFold's uniref.a3m.
    downloaded = output_dir / "unpaired" / "0.a3m"
    if not downloaded.is_file():
        return
    _validate_a3m_query(downloaded, sequence)
    if _alignment_depth(downloaded) > 1:
        result_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, unpaired)
        (result_dir / "pairing.a3m").write_text(f">query\n{sequence}\n", encoding="utf-8")


def _run_protenix_search(
    sequences: Sequence[str], output_dir: str, mode: str
) -> Sequence[str]:
    environment = dict(os.environ)
    environment.setdefault("MMSEQS_SERVICE_HOST_URL", DEFAULT_MSA_SERVER_URL)
    script = (
        "import json,sys; "
        "from runner.msa_search import msa_search; "
        "p=json.load(sys.stdin); "
        "r=list(msa_search(p['sequences'],p['output_dir'],mode=p['mode'])); "
        "print('\\n__PROTENIX_MSA_RESULT__'+json.dumps(r))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(
                {
                    "sequences": list(sequences),
                    "output_dir": output_dir,
                    "mode": mode,
                }
            ),
            text=True,
            capture_output=True,
            check=False,
            env=environment,
            timeout=float(os.environ.get("OPENDDE_HARNESS_MSA_SEARCH_TIMEOUT", "1800")),
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalServiceUnavailableError(
            MSA_SERVICE, "the online MSA search timed out", environment["MMSEQS_SERVICE_HOST_URL"]
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        reason = connection_reason(RuntimeError(detail)) if detail else None
        if reason is not None:
            raise ExternalServiceUnavailableError(MSA_SERVICE, reason, environment["MMSEQS_SERVICE_HOST_URL"])
        raise RuntimeError(
            "Protenix online MSA search failed"
            + (f": {detail}" if detail else "")
        )
    marker = "__PROTENIX_MSA_RESULT__"
    if marker not in completed.stdout:
        raise RuntimeError("Protenix online MSA search returned no result manifest")
    result = json.loads(completed.stdout.rsplit(marker, 1)[1])
    if not isinstance(result, list):
        raise RuntimeError("Protenix online MSA search returned an invalid result manifest")
    if mode == "protenix" and len(sequences) == len(result) == 1:
        _normalize_protenix_result(Path(output_dir), Path(result[0]), sequences[0])
    return [str(path) for path in result]


def search_target_msa(
    *,
    target_name: str,
    chain_id: str,
    sequence: str,
    output_root: Path,
    force: bool = False,
    search: Callable[[Sequence[str], str, str], Sequence[str]] | None = None,
) -> dict[str, object]:
    """Search one antigen chain and retain paths readable by this compute worker."""

    normalized_sequence = _normalize_sequence(sequence)
    normalized_chain = _safe_name(chain_id)
    digest = hashlib.sha256(normalized_sequence.encode()).hexdigest()
    destination = output_root / f"{_safe_name(target_name)}-{normalized_chain}-{digest[:12]}"
    unpaired = destination / "non_pairing.a3m"
    paired = destination / "pairing.a3m"
    server_url = os.environ.get("MMSEQS_SERVICE_HOST_URL", DEFAULT_MSA_SERVER_URL).rstrip("/")
    server_mode = os.environ.get("OPENDDE_HARNESS_MSA_SERVER_MODE", "protenix").strip().lower()
    if server_mode not in {"protenix", "colabfold"}:
        raise ValueError(
            "OPENDDE_HARNESS_MSA_SERVER_MODE must be 'protenix' or 'colabfold'"
        )

    if not force and unpaired.is_file() and paired.is_file():
        _validate_a3m_query(unpaired, normalized_sequence)
        _validate_a3m_query(paired, normalized_sequence)
        depth = _alignment_depth(unpaired)
        if depth > 1:
            return {
                "target_name": target_name,
                "chain_id": chain_id,
                "sequence_sha256": digest,
                "unpaired_msa_path": str(unpaired.resolve()),
                "paired_msa_path": str(paired.resolve()),
                "alignment_depth": depth,
                "cached": True,
                "server_url": server_url,
                "server_mode": server_mode,
            }

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="target-msa-", dir=output_root))
    try:
        result_dirs = (search or _run_protenix_search)(
            [normalized_sequence], str(temporary), server_mode
        )
        if len(result_dirs) != 1:
            raise RuntimeError("Protenix MSA service returned an unexpected result count")
        result_dir = Path(result_dirs[0])
        generated_unpaired = result_dir / "non_pairing.a3m"
        generated_paired = result_dir / "pairing.a3m"
        if not generated_unpaired.is_file() or not generated_paired.is_file():
            raise RuntimeError(
                "Protenix MSA service did not produce paired and unpaired A3M files"
            )
        _validate_a3m_query(generated_unpaired, normalized_sequence)
        _validate_a3m_query(generated_paired, normalized_sequence)
        depth = _alignment_depth(generated_unpaired)
        if depth <= 1:
            raise RuntimeError(
                "online MSA search returned only the query sequence; no homologous target MSA was found"
            )
        staged = temporary / "final"
        staged.mkdir()
        shutil.copy2(generated_unpaired, staged / "non_pairing.a3m")
        shutil.copy2(generated_paired, staged / "pairing.a3m")
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    return {
        "target_name": target_name,
        "chain_id": chain_id,
        "sequence_sha256": digest,
        "unpaired_msa_path": str(unpaired.resolve()),
        "paired_msa_path": str(paired.resolve()),
        "alignment_depth": depth,
        "cached": False,
        "server_url": server_url,
        "server_mode": server_mode,
    }

"""
ProTrek protein structure and sequence search tools for Agno.

Provides tools for:
- Sequence-based protein function search
- Structure-based protein function search
"""

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from opendde_harness.plugin.protein_design.core.external import (
    DEFAULT_PROTREK_URL,
    PROTREK_SERVICE,
    ExternalServiceUnavailableError,
    connection_reason,
    protrek_endpoint,
)

# Fix gradio temp directory permission issue
_gradio_temp = os.path.join(tempfile.gettempdir(), f"gradio_{os.environ.get('USER', 'agent')}")
os.makedirs(_gradio_temp, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", _gradio_temp)

try:
    from gradio_client import Client, handle_file

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

PROTREK_DISABLED_MESSAGE = "ProTrek search is disabled; set PROTREK_ENDPOINT to enable it."
DEFAULT_NPROBE = 10000
DEFAULT_TOPK = 5
DEFAULT_DB = "Swiss-Prot"
DEFAULT_SUBSECTION_TYPE = "Function"
MAX_SEARCH_RESULTS = 5
DEFAULT_PROTREK_TIMEOUT_SECONDS = 10.0


class ProtrekClient:
    """ProTrek API client for protein structure and sequence search."""

    def __init__(
        self,
        url: Optional[str] = None,
        default_nprobe: int = DEFAULT_NPROBE,
        default_topk: int = DEFAULT_TOPK,
        default_db: str = DEFAULT_DB,
        default_subsection_type: str = DEFAULT_SUBSECTION_TYPE,
        timeout_seconds: float = DEFAULT_PROTREK_TIMEOUT_SECONDS,
    ):
        self.url = url or protrek_endpoint() or DEFAULT_PROTREK_URL
        self.default_nprobe = default_nprobe
        self.default_topk = default_topk
        self.default_db = default_db
        self.default_subsection_type = default_subsection_type
        self.timeout_seconds = float(timeout_seconds)

        if GRADIO_AVAILABLE:
            try:
                self._client = Client(
                    self.url,
                    httpx_kwargs={"timeout": self.timeout_seconds},
                    verbose=False,
                )
            except Exception as exc:
                raise _unavailable(exc, self.url) from exc
        else:
            self._client = None

    def search_by_sequence(
        self,
        sequence: str,
        query_type: str = "text",
        subsection_type: Optional[str] = None,
        db: Optional[str] = None,
        nprobe: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Any:
        """Search for proteins by amino acid sequence."""
        if self._client is None:
            return None

        subsection_type = subsection_type or self.default_subsection_type
        db = db or self.default_db
        nprobe = nprobe or self.default_nprobe
        topk = topk or self.default_topk

        return self._client.predict(
            input=sequence,
            input_type="sequence",
            query_type=query_type,
            subsection_type=subsection_type,
            db=db,
            nprobe=nprobe,
            topk=topk,
            api_name="/search",
        )

    def parse_pdb_file(self, pdb_file_path: Union[str, Path], chain: str = "A") -> Any:
        """Parse a PDB file and extract structure information."""
        if self._client is None:
            return None

        return self._client.predict(
            file=handle_file(str(pdb_file_path)), input_type="structure", chain=chain, api_name="/parse_pdb_file"
        )

    def search_by_structure(
        self,
        structure_result: Any,
        query_type: str = "text",
        subsection_type: Optional[str] = None,
        db: Optional[str] = None,
        nprobe: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Any:
        """Search for proteins using parsed structure information."""
        if self._client is None:
            return None

        subsection_type = subsection_type or self.default_subsection_type
        db = db or self.default_db
        nprobe = nprobe or self.default_nprobe
        topk = topk or self.default_topk

        return self._client.predict(
            input=structure_result,
            input_type="structure",
            query_type=query_type,
            subsection_type=subsection_type,
            db=db,
            nprobe=nprobe,
            topk=topk,
            api_name="/search",
        )

    def search_by_structure_file(
        self,
        pdb_file_path: Union[str, Path],
        chain: str = "A",
        query_type: str = "text",
        subsection_type: Optional[str] = None,
        db: Optional[str] = None,
        nprobe: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Any:
        """Parse a PDB file and search in one step."""
        structure_result = self.parse_pdb_file(pdb_file_path, chain)
        if structure_result is None:
            return None
        return self.search_by_structure(structure_result, query_type, subsection_type, db, nprobe, topk)


# Global client
_protrek_client = None


def _unavailable(exc: BaseException, endpoint: str | None) -> ExternalServiceUnavailableError:
    """Classify one ProTrek failure; connection-level ones stay optional."""
    if isinstance(exc, ExternalServiceUnavailableError):
        return exc
    reason = connection_reason(exc)
    if reason is None:
        raise exc
    return ExternalServiceUnavailableError(PROTREK_SERVICE, reason, endpoint)


def get_protrek_client() -> ProtrekClient:
    """Get or create the global ProTrek client instance."""
    global _protrek_client
    if _protrek_client is None:
        _protrek_client = ProtrekClient()
    return _protrek_client


def _parse_protrek_result(protrek_result: Any) -> List[Dict[str, Any]]:
    """Parse ProTrek search result from Gradio client format."""
    parsed_results = []

    try:
        if isinstance(protrek_result, dict):
            value_dict = protrek_result.get("value", protrek_result)
        else:
            return parsed_results

        data = value_dict.get("data", [])
        headers = [str(header).strip().lower().replace(" ", "_") for header in value_dict.get("headers", [])]

        for entry in data:
            if not isinstance(entry, list) or not entry:
                continue
            row = {
                headers[index] if index < len(headers) else f"field_{index}": value for index, value in enumerate(entry)
            }
            score = row.get("matching_score")
            if not isinstance(score, (int, float)):
                continue
            parsed_results.append(
                {
                    "id": str(row.get("id", "")),
                    "description": str(row.get("description", row.get("id", ""))),
                    "sequence": str(row.get("sequence", "")),
                    "foldseek_sequence": str(row.get("foldseek_sequence", "")),
                    "length": row.get("length"),
                    "sequence_identity": str(row.get("sequence_identity", "")),
                    "matching_score": float(score),
                }
            )
    except Exception:
        return []

    return parsed_results


def _format_protrek_summary(
    parsed_results: List[Dict[str, Any]], max_results: int = 5, source_type: str = "sequence"
) -> str:
    """Format parsed ProTrek results into a markdown summary."""
    if not parsed_results:
        return ""

    source_desc = "structure-based search" if source_type.lower() == "structure" else "sequence-based search"
    title = (
        "### ProTrek Structure-Based Search Results"
        if source_type.lower() == "structure"
        else "### ProTrek Sequence-Based Search Results"
    )

    lines = [
        title,
        "",
        f"The following homolog evidence was found via {source_desc}:",
        "Sequence homology supports conservation or natural-variation hypotheses only; "
        "it does not by itself support target-specific affinity or contact mechanisms.",
        "",
    ]

    sorted_results = sorted(parsed_results, key=lambda x: x["matching_score"], reverse=True)
    top_results = sorted_results[: min(max(1, int(max_results)), MAX_SEARCH_RESULTS)]

    for i, result in enumerate(top_results, 1):
        hit_id = result.get("id") or result.get("description") or "unknown"
        details = [f"score={result['matching_score']:.3f}"]
        if result.get("sequence_identity"):
            details.append(f"identity={result['sequence_identity']}")
        if result.get("aligned_identity") is not None:
            details.append(f"aligned_identity={result['aligned_identity']:.1f}%")
        if result.get("length") is not None:
            details.append(f"length={result['length']}")
        lines.append(f"{i}. **{hit_id}** ({', '.join(details)})")
        if result.get("sequence"):
            lines.append(f"   - sequence: `{result['sequence']}`")
        elif result.get("foldseek_sequence"):
            lines.append(f"   - Foldseek sequence: `{result['foldseek_sequence']}`")
        if result.get("query_aligned_alternatives"):
            lines.append(
                "   - query-aligned natural alternatives (0-based): "
                + ", ".join(result["query_aligned_alternatives"][:40])
            )

    lines.append("")
    return "\n".join(lines)


def _annotate_sequence_alignments(query_sequence: str, parsed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use Biotite global alignment to ground homolog differences in query positions."""
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import SubstitutionMatrix, align_optimal

    query = ProteinSequence(query_sequence)
    matrix = SubstitutionMatrix.std_protein_matrix()
    annotated: List[Dict[str, Any]] = []
    for result in parsed_results:
        hit_sequence = str(result.get("sequence") or "")
        if not hit_sequence:
            annotated.append(result)
            continue
        try:
            hit = ProteinSequence(hit_sequence)
            alignment = align_optimal(
                query,
                hit,
                matrix,
                gap_penalty=(-10, -1),
                terminal_penalty=False,
                max_number=1,
            )[0]
            aligned = 0
            identical = 0
            alternatives: List[str] = []
            for query_index, hit_index in alignment.trace:
                if query_index < 0 or hit_index < 0:
                    continue
                aligned += 1
                query_residue = query_sequence[int(query_index)]
                hit_residue = hit_sequence[int(hit_index)]
                if query_residue == hit_residue:
                    identical += 1
                elif query_residue != "X":
                    alternatives.append(f"{query_residue}{int(query_index)}{hit_residue}")
            annotated.append(
                {
                    **result,
                    "aligned_identity": 100.0 * identical / aligned if aligned else 0.0,
                    "query_aligned_alternatives": alternatives,
                }
            )
        except Exception:
            annotated.append(result)
    return annotated


def _looks_like_antibody_domain(sequence: str) -> bool:
    """Conservative safeguard against transferring motifs from unrelated folds."""
    sequence = "".join(str(sequence).split()).upper()
    if not 90 <= len(sequence) <= 150 or sequence.count("C") < 2:
        return False
    first_cys = sequence.find("C")
    second_cys = sequence.find("C", first_cys + 1)
    return "VQL" in sequence[:15] and 15 <= first_cys <= 35 and second_cys >= 75


def _prepare_protrek_structure_path(structure_path: Path) -> Path:
    """Convert mmCIF to PDB because ProTrek's parser only handles PDB reliably."""
    if structure_path.suffix.lower() not in {".cif", ".mmcif"}:
        return structure_path
    stat = structure_path.stat()
    digest = hashlib.sha256(f"{structure_path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:16]
    converted = Path(_gradio_temp) / f"protrek_{digest}.pdb"
    try:
        import numpy as np
        from biotite.structure.io import load_structure, save_structure

        atoms = load_structure(
            structure_path,
            model=1,
            extra_fields=["b_factor", "occupancy"],
        )
        # Predicted mmCIFs may omit these optional atom_site columns.
        # Biotite correctly parses the coordinates and chain IDs; normalize only
        # the missing PDB presentation fields before serialization.
        atoms.b_factor = np.nan_to_num(atoms.b_factor, nan=0.0, posinf=0.0, neginf=0.0)
        atoms.occupancy = np.nan_to_num(atoms.occupancy, nan=1.0, posinf=1.0, neginf=1.0)
        temporary = converted.with_suffix(".tmp.pdb")
        save_structure(temporary, atoms)
        temporary.replace(converted)
    except Exception as exc:
        raise ValueError(f"Could not convert mmCIF for ProTrek: {exc}") from exc
    return converted


def search_protrek_sequence(
    sequence: str,
    query_type: str = "sequence",
    subsection_type: str = "Function",
    db: str = "Swiss-Prot",
    nprobe: int = 10000,
    topk: int = 5,
) -> str:
    """
    Search for protein functions based on amino acid sequence.

    Uses the ProTrek API to find functional annotations and related database entries
    for a given protein sequence.

    Args:
        sequence: The amino acid sequence to search for.
        query_type: Type of query result (default: "sequence").
        subsection_type: Subsection type for text descriptions (default: "Function").
        db: Target database to search against (default: "Swiss-Prot").
        nprobe: Number of search clusters to probe (default: 10000).
        topk: Number of top results to return (default: 5, capped at 5).

    Returns:
        A formatted string report of matching protein functions.
    """
    if isinstance(sequence, dict):
        seq_str = "".join(sequence.values())
    else:
        seq_str = sequence
    seq_str = "".join(str(seq_str).split()).upper()
    if not seq_str or any(residue not in "ACDEFGHIKLMNPQRSTVWYX" for residue in seq_str):
        return "Invalid protein sequence; use one full protein chain with amino-acid letters only."
    if protrek_endpoint() is None:
        return PROTREK_DISABLED_MESSAGE
    topk = min(max(1, int(topk)), MAX_SEARCH_RESULTS)
    return _cached_search_protrek_sequence(seq_str, query_type, subsection_type, db, int(nprobe), topk)


@lru_cache(maxsize=128)
def _cached_search_protrek_sequence(
    seq_str: str,
    query_type: str,
    subsection_type: str,
    db: str,
    nprobe: int,
    topk: int,
) -> str:
    """Cached implementation kept private so Agno sees a normal tool function."""
    parsed = list(_cached_search_protrek_sequence_records(seq_str, query_type, subsection_type, db, nprobe, topk))
    if not parsed and get_protrek_client()._client is None:
        return "ProTrek service unavailable. Please check gradio_client installation."

    if parsed:
        return _format_protrek_summary(parsed, max_results=topk, source_type="sequence")

    return "No results found."


@lru_cache(maxsize=128)
def _cached_search_protrek_sequence_records(
    seq_str: str,
    query_type: str,
    subsection_type: str,
    db: str,
    nprobe: int,
    topk: int,
) -> tuple[Dict[str, Any], ...]:
    """Cached structured sequence search shared by text and proposal paths."""
    client = get_protrek_client()
    try:
        search_result = client.search_by_sequence(
            sequence=seq_str,
            query_type=query_type,
            subsection_type=subsection_type,
            db=db,
            nprobe=nprobe,
            topk=topk,
        )
    except Exception as exc:
        raise _unavailable(exc, client.url) from exc
    if search_result is None:
        return ()
    result_data = (
        search_result[3] if isinstance(search_result, (list, tuple)) and len(search_result) > 3 else search_result
    )
    parsed = _annotate_sequence_alignments(seq_str, _parse_protrek_result(result_data))
    return tuple(dict(hit) for hit in parsed[:topk])


def search_protrek_structure(
    pdb_file_path: str,
    chain: str = "A",
    query_type: str = "sequence",
    subsection_type: str = "Function",
    db: str = "Swiss-Prot",
    nprobe: int = 10000,
    topk: int = 5,
) -> str:
    """
    Search for protein functions based on 3D structure file.

    Uses the ProTrek API to find functional annotations for a protein
    based on its 3D structure (PDB or CIF format).

    Args:
        pdb_file_path: Path to the PDB or CIF structure file.
        chain: Chain identifier to extract (default: "A").
        query_type: Type of query result (default: "text").
        subsection_type: Subsection type for text descriptions (default: "Function").
        db: Target database to search against (default: "Swiss-Prot").
        nprobe: Number of search clusters to probe (default: 10000).
        topk: Number of top results to return (default: 10).

    Returns:
        A formatted string report of matching protein functions.
    """
    structure_path = Path(pdb_file_path).expanduser().resolve()
    if not structure_path.is_file():
        return f"Structure file not found: {structure_path}"
    if protrek_endpoint() is None:
        return PROTREK_DISABLED_MESSAGE
    try:
        search_path = _prepare_protrek_structure_path(structure_path)
    except Exception as exc:
        return f"ProTrek structure search unavailable: {exc}"
    topk = min(max(1, int(topk)), MAX_SEARCH_RESULTS)
    try:
        return _cached_search_protrek_structure(
            str(search_path), chain, query_type, subsection_type, db, int(nprobe), topk
        )
    finally:
        if search_path != structure_path:
            search_path.unlink(missing_ok=True)


@lru_cache(maxsize=64)
def _cached_search_protrek_structure_records(
    pdb_file_path: str,
    chain: str,
    query_type: str,
    subsection_type: str,
    db: str,
    nprobe: int,
    topk: int,
) -> tuple[Dict[str, Any], ...]:
    client = get_protrek_client()
    try:
        search_result = client.search_by_structure_file(
            pdb_file_path=pdb_file_path,
            chain=chain,
            query_type=query_type,
            subsection_type=subsection_type,
            db=db,
            nprobe=nprobe,
            topk=topk,
        )
    except Exception as exc:
        raise _unavailable(exc, client.url) from exc
    if search_result is None:
        return ()
    result_data = (
        search_result[3] if isinstance(search_result, (list, tuple)) and len(search_result) > 3 else search_result
    )
    parsed = [hit for hit in _parse_protrek_result(result_data) if _looks_like_antibody_domain(hit.get("sequence", ""))]
    return tuple(dict(hit) for hit in parsed[:topk])


@lru_cache(maxsize=64)
def _cached_search_protrek_structure(
    pdb_file_path: str,
    chain: str,
    query_type: str,
    subsection_type: str,
    db: str,
    nprobe: int,
    topk: int,
) -> str:
    """Cached implementation kept private so Agno can register the public tool."""
    try:
        parsed = list(
            _cached_search_protrek_structure_records(
                pdb_file_path, chain, query_type, subsection_type, db, nprobe, topk
            )
        )
    except ExternalServiceUnavailableError:
        raise
    except Exception as exc:
        return f"ProTrek structure search unavailable: {exc}"

    if parsed:
        return _format_protrek_summary(parsed, max_results=topk, source_type="structure")

    return "No antibody-like structural analogs passed the domain filter."

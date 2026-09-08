#!/usr/bin/env python3
"""
Epitope analysis backend - batch processing.

Analyze multiple structures for epitope identification.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .epitope_analysis import (
    analyze_epitope,
    load_cdr_regions_from_yaml,
    parse_hotspots,
)


def find_structure_files(structure_dir: str, pattern: str = "*.pdb") -> List[Path]:
    """Find all structure files in directory."""
    path = Path(structure_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {structure_dir}")

    # Try PDB first, then CIF
    structures = list(path.glob(pattern))
    if not structures and pattern == "*.pdb":
        structures = list(path.glob("*.cif"))

    return sorted(structures)


def process_batch(
    structure_files: List[Path],
    antibody_chains: List[str],
    antigen_chains: List[str],
    cdr_regions: Dict[str, Dict[str, set[int]]],
    cutoff: float = 4.5,
    hotspots: List = None,
    workers: int = 4,
) -> List[Dict[str, Any]]:
    """Process multiple structures in parallel."""
    results = []

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}

        for structure_file in structure_files:
            future = executor.submit(
                analyze_epitope,
                structure_file=str(structure_file),
                antibody_chains=antibody_chains,
                antigen_chains=antigen_chains,
                cdr_regions=cdr_regions,
                cutoff=cutoff,
                hotspots=hotspots,
            )
            futures[future] = structure_file.name

        # Collect results
        for future in cf.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✓ Completed: {name}", file=sys.stderr)
            except Exception as e:
                print(f"✗ Failed: {name} - {e}", file=sys.stderr)
                results.append(
                    {
                        "structure": name,
                        "error": str(e),
                        "epitope_size": None,
                        "total_contacts": None,
                    }
                )

    return results


def write_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Write results to CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "structure",
        "epitope_size",
        "total_contacts",
        "hotspot_coverage",
        "contacted_hotspots",
        "missed_hotspots",
        "cdr3_h_contacts",
        "cdr2_h_contacts",
        "cdr1_h_contacts",
        "error",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            if "error" in result and result.get("epitope_size") is None:
                # Error case
                row = {
                    "structure": result["structure"],
                    "error": result["error"],
                }
            else:
                # Success case
                hotspot = result.get("hotspot_analysis", {})
                cdrs = result.get("cdr_contributions", {})

                row = {
                    "structure": result["structure"],
                    "epitope_size": result.get("epitope_size", 0),
                    "total_contacts": result.get("total_contacts", 0),
                    "hotspot_coverage": hotspot.get("hotspot_coverage", ""),
                    "contacted_hotspots": hotspot.get("contacted_hotspots", ""),
                    "missed_hotspots": ";".join(hotspot.get("missed_hotspots", [])),
                    "cdr3_h_contacts": cdrs.get("CDR3_H", {}).get("total_contacts", ""),
                    "cdr2_h_contacts": cdrs.get("CDR2_H", {}).get("total_contacts", ""),
                    "cdr1_h_contacts": cdrs.get("CDR1_H", {}).get("total_contacts", ""),
                    "error": "",
                }

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Batch epitope analysis for multiple structures")
    parser.add_argument("--structure_dir", required=True, help="Directory containing structure files")
    parser.add_argument("--antibody_chains", required=True, help="Comma-separated antibody chain IDs (e.g., H,L)")
    parser.add_argument("--antigen_chains", required=True, help="Comma-separated antigen chain IDs (e.g., A)")
    parser.add_argument(
        "--config_yaml", required=True, help="Design YAML containing the binder CDR/fixed-residue configuration"
    )
    parser.add_argument("--binder_name", help="Initial binder name when YAML binders use different CDR configurations")
    parser.add_argument("--pattern", default="*.pdb", help="File pattern to match (default: *.pdb)")
    parser.add_argument("--cutoff", type=float, default=4.5, help="Distance cutoff in Angstroms (default: 4.5)")
    parser.add_argument("--hotspots", help="Comma-separated known hotspots (e.g., A:45,A:52)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--output_json", help="Optional: Also write detailed JSON results")

    args = parser.parse_args()

    try:
        # Find structure files
        structure_files = find_structure_files(args.structure_dir, args.pattern)
        print(f"Found {len(structure_files)} structure files", file=sys.stderr)

        if not structure_files:
            print(f"No structure files found in {args.structure_dir}", file=sys.stderr)
            sys.exit(1)

        # Parse inputs
        antibody_chains = [c.strip() for c in args.antibody_chains.split(",")]
        antigen_chains = [c.strip() for c in args.antigen_chains.split(",")]
        hotspots = parse_hotspots(args.hotspots) if args.hotspots else None
        cdr_regions = load_cdr_regions_from_yaml(args.config_yaml, binder_name=args.binder_name)

        # Process batch
        print(f"Processing with {args.workers} workers...", file=sys.stderr)
        results = process_batch(
            structure_files,
            antibody_chains,
            antigen_chains,
            cdr_regions,
            cutoff=args.cutoff,
            hotspots=hotspots,
            workers=args.workers,
        )

        # Write CSV
        write_results_csv(results, args.output_csv)
        print(f"\n✅ CSV results written to {args.output_csv}", file=sys.stderr)

        # Optionally write JSON
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(results, indent=2))
            print(f"✅ JSON results written to {args.output_json}", file=sys.stderr)

        # Summary
        successful = sum(1 for r in results if "error" not in r or not r.get("error"))
        failed = len(results) - successful
        print(f"\nSummary: {successful} succeeded, {failed} failed", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

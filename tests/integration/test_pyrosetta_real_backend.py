"""Opt-in scientific smoke test; uses a user-supplied protein complex on the compute host."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.servers.backends import pyrosetta_worker
from opendde_harness.plugin.protein_design.servers.backends.pyrosetta_analysis import (
    PYROSETTA_METRICS,
    PyRosettaConfig,
    analyze_batch,
)

pytestmark = pytest.mark.integration


def test_real_fastrelax_and_interface_analyzer(tmp_path):
    if importlib.util.find_spec("pyrosetta") is None:
        pytest.skip("PyRosetta is not installed")
    source = os.environ.get("OPENDDE_TEST_PYROSETTA_PDB")
    chain_config = os.environ.get("OPENDDE_TEST_PYROSETTA_CHAINS")
    if not source or not chain_config:
        pytest.skip("Set OPENDDE_TEST_PYROSETTA_PDB and OPENDDE_TEST_PYROSETTA_CHAINS for a real complex")
    chains = json.loads(chain_config)
    before = Path(source).read_bytes()
    results = analyze_batch(
        [str(Path(source).resolve())] * 2,
        [chains["sequences"]] * 2,
        binder_chains=chains["binder"],
        target_chains=chains["target"],
        config=PyRosettaConfig(enabled=True, max_workers=2, relax_repeats=1, max_iter=50, timeout_seconds=900.0),
        output_dir=tmp_path,
    )
    assert len(results) == 2
    for result in results:
        assert result.status == "success", result.error
        assert set(result.metrics) == set(PYROSETTA_METRICS)
        assert result.metrics["rosetta_interface_sasa"] > 0
        assert result.metrics["rosetta_interface_dg_per_sasa"] == pytest.approx(
            100.0 * result.metrics["rosetta_interface_dg"] / result.metrics["rosetta_interface_sasa"]
        )
        assert result.provenance["steps"] == ["FastRelax", "InterfaceAnalyzer"]
        assert Path(result.provenance["python_executable"]).resolve() == Path(sys.executable).resolve()
        assert Path(result.provenance["worker_module_path"]).resolve() == Path(pyrosetta_worker.__file__).resolve()
        assert isinstance(result.provenance["pid"], int)
        assert result.provenance["pid"] > 0
        assert result.provenance["pid"] != os.getpid()
        assert result.contact_residues
        assert {row.partner for row in result.contact_residues} == {"binder", "target"}
        assert all(row.min_partner_distance <= 5.0 for row in result.contact_residues)
        for row in result.contact_residues:
            assert chains["sequences"][row.chain_id][row.residue_index] == row.amino_acid
            assert row.bound_score_reu - row.separated_score_reu == pytest.approx(row.interface_dg_reu)
        assert Path(result.relaxed_structure_path).is_file()
    assert Path(source).read_bytes() == before
    assert results[0].provenance["pid"] != results[1].provenance["pid"]
    assert results[0].relaxed_structure_path != results[1].relaxed_structure_path
    assert results[0].metrics == pytest.approx(results[1].metrics)

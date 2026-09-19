"""Scientific loss traces preserve calibration without relaxing metadata safety."""

import json

from opendde_harness.plugin.protein_design.core.contracts import Candidate
from opendde_harness.plugin.protein_design.core.tracing import _safe_metadata, build_cycle_artifact
from opendde_harness.plugin.protein_design.servers.backends.loss_objective import (
    DEFAULT_LOSS_WEIGHTS,
    calculate_loss_objective,
)


def bounded_loss():
    weights = {name: 0.0 for name in DEFAULT_LOSS_WEIGHTS}
    weights.update(plddt=1.0, esm2=0.1)
    return calculate_loss_objective(
        {"plddt": 0.2},
        -2.0,
        weights=weights,
        metric_values={"rosetta_interface_dg": -30.0},
        metric_terms={"rosetta_interface_dg": {"weight": 0.5, "direction": "minimize"}},
        loss_combination={
            "mode": "bounded_grouped",
            "calibration_id": "trace-test-fixture-not-scientific-defaults",
            "groups": {
                "confidence": {"budget": 0.4, "terms": ["plddt"]},
                "sequence": {"budget": 0.2, "terms": ["esm2"]},
                "interface": {"budget": 0.4, "terms": ["rosetta_interface_dg"]},
            },
            "anchors": {
                "plddt": {"good": 0.0, "bad": 1.0},
                "esm2": {"good": 0.0, "bad": -5.0},
                "rosetta_interface_dg": {"good": -60.0, "bad": 0.0},
            },
        },
    )


def test_cycle_artifact_preserves_complete_fixed_loss_calibration_and_group_membership():
    loss = bounded_loss()
    candidate = Candidate(
        candidate_id="scored",
        sequence="AAA",
        objective=loss["loss"],
        metrics={"loss": loss["loss"]},
        metadata={"success": True, "loss": loss},
    )
    artifact = build_cycle_artifact(
        task_id="bounded-trace",
        target="target",
        cycle=0,
        objective_key="loss",
        minimize=True,
        candidates=[candidate],
        cycle_best_id="scored",
        global_best_id="scored",
        known_candidate_ids={"scored"},
        structure_artifacts={},
    )
    saved = artifact["candidates"][0]["metadata"]["loss"]
    assert saved == loss
    assert saved["loss_combination"]["groups"]["interface"]["terms"] == ["rosetta_interface_dg"]
    assert saved["loss_combination"]["anchors"]["rosetta_interface_dg"] == {"good": -60.0, "bad": 0.0}
    assert saved["groups"]["interface"]["terms"] == ["rosetta_interface_dg"]
    assert saved["legacy_loss_breakdown"]["components"]["rosetta_interface_dg"]["raw"] == -30.0


def test_deeper_loss_capture_still_redacts_secrets_at_every_level():
    loss = bounded_loss()
    loss["api_key"] = "SENSITIVE"
    loss["loss_combination"]["authorization"] = "SENSITIVE"
    loss["loss_combination"]["groups"]["interface"]["secret"] = "SENSITIVE"
    loss["loss_combination"]["anchors"]["rosetta_interface_dg"]["password"] = "SENSITIVE"
    loss["components"]["rosetta_interface_dg"]["prompt"] = "SENSITIVE"
    loss["legacy_loss_breakdown"]["components"]["rosetta_interface_dg"]["token"] = "SENSITIVE"
    safe = _safe_metadata({"loss": loss, "authorization": "SENSITIVE"})
    assert "SENSITIVE" not in json.dumps(safe)
    assert safe["loss"]["loss_combination"]["groups"]["interface"]["terms"] == ["rosetta_interface_dg"]


def test_deeper_loss_capture_retains_item_text_finite_and_depth_limits():
    value = {
        "rows": list(range(100)),
        "labels": {str(i): i for i in range(100)},
        "text": "x" * 600,
        "nan": float("nan"),
        "infinity": float("inf"),
        "deep": {"a": {"b": {"c": {"d": {"e": {"beyond_limit": 1}}}}}},
    }
    safe = _safe_metadata({"loss": value, "unrelated": value})
    for category in ("loss", "unrelated"):
        assert len(safe[category]["rows"]) == 64
        assert len(safe[category]["labels"]) == 64
        assert len(safe[category]["text"]) == 512
        assert "nan" not in safe[category]
        assert "infinity" not in safe[category]
        assert "beyond_limit" not in json.dumps(safe[category])
    assert len(_safe_metadata({"loss": {str(i): i for i in range(100)}})["loss"]) == 64
    assert len(_safe_metadata({str(i): i for i in range(100)})) == 64


def test_extra_depth_is_scoped_to_loss_not_arbitrary_metadata():
    nested = {"combination": {"groups": {"interface": {"terms": ["rosetta_interface_dg"]}}}}
    safe = _safe_metadata({"loss": nested, "unrelated": nested})
    assert safe["loss"] == nested
    assert "unrelated" not in safe


def test_raw_min_ipae_and_its_chain_mapping_are_preserved_without_using_loss_terms():
    evidence = {
        "status": "success",
        "value": 4.23456789,
        "units": "angstrom",
        "binder_chains": ["B"],
        "target_chains": ["A"],
        "directional_minima": {"binder_to_target": 5.1, "target_to_binder": 4.23456789},
        "chain_mapping": {
            "A": {"asym_id": 0, "residue_count": 10, "valid_frame_count": 10},
            "B": {"asym_id": 1, "residue_count": 5, "valid_frame_count": 5},
        },
        "frame_policy": "Require token_has_frame=1 on both the PAE row and column",
        "matrix_shape": [15, 15],
    }
    candidate = Candidate(
        candidate_id="raw-pae",
        sequence="AAA",
        objective=0.4,
        metrics={"loss": 0.4, "min_ipae": 4.23456789},
        metadata={"min_ipae": evidence, "loss": {"loss_components": {"i_pae": 0.2}}},
    )
    artifact = build_cycle_artifact(
        task_id="raw-pae-trace",
        target="target",
        cycle=0,
        objective_key="loss",
        minimize=True,
        candidates=[candidate],
        cycle_best_id="raw-pae",
        global_best_id="raw-pae",
        known_candidate_ids={"raw-pae"},
        structure_artifacts={},
    )
    saved = artifact["candidates"][0]
    assert saved["metrics"]["min_ipae"] == 4.23456789
    assert saved["metadata"]["min_ipae"] == evidence
    assert saved["metadata"]["loss"]["loss_components"]["i_pae"] == 0.2

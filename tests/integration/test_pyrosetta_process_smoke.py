"""Exercise real child processes without requiring licensed PyRosetta binaries."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from opendde_harness.plugin.protein_design.servers.backends import pyrosetta_analysis as analysis

pytestmark = pytest.mark.integration

CHILD = """
import json, os, pathlib, sys, time
request = json.load(sys.stdin)
output = pathlib.Path(request['output_dir'])
(output / 'pid').write_text(str(os.getpid()))
mode = request['structure_path']
if mode == 'timeout':
    time.sleep(60)
if mode == 'crash':
    sys.exit(7)
if mode == 'parallel':
    (output / 'ready').touch()
    deadline = time.monotonic() + 10
    while len(list(output.parent.glob('*/ready'))) < 2:
        if time.monotonic() > deadline:
            sys.exit(8)
        time.sleep(0.01)
path = output / 'relaxed.pdb'
path.write_text('END\\n')
metrics = {name: 1.0 for name in request['sequences']}
(output / 'analysis.json').write_text(json.dumps({
    'status': 'success', 'metrics': metrics, 'relaxed_structure_path': str(path),
    'provenance': {'pid': os.getpid(), 'omp_threads': os.environ.get('OMP_NUM_THREADS')}
}))
"""


@pytest.fixture
def child_backend(monkeypatch):
    run = subprocess.run

    def substitute(command, **kwargs):
        assert command[1:3] == ["-m", "opendde_harness.plugin.protein_design.servers.backends.pyrosetta_worker"]
        return run([sys.executable, "-c", CHILD], **kwargs)

    monkeypatch.setattr(analysis.subprocess, "run", substitute)


def test_candidates_execute_in_distinct_overlapping_cpu_processes(tmp_path, monkeypatch, child_backend):
    monkeypatch.setattr(analysis, "cpu_workers", lambda *_: 2)
    results = analysis.analyze_batch(
        ["parallel", "parallel"],
        [dict.fromkeys(analysis.PYROSETTA_METRICS, "A")] * 2,
        binder_chains=["B"],
        target_chains=["A"],
        config=analysis.PyRosettaConfig(enabled=True, max_workers=2, timeout_seconds=15.0),
        output_dir=tmp_path,
    )
    assert [result.status for result in results] == ["success", "success"]
    assert len({result.provenance["pid"] for result in results}) == 2
    assert all(result.provenance["pid"] != os.getpid() for result in results)
    assert all(result.provenance["omp_threads"] == "1" for result in results)


@pytest.mark.parametrize("mode", ["timeout", "crash"])
def test_failed_child_is_reaped_and_returns_no_metrics(tmp_path, child_backend, mode):
    output = tmp_path / "child"
    result = analysis._analyze_one(mode, {}, "B_A", analysis.PyRosettaConfig(enabled=True, timeout_seconds=1.0), output)
    assert result.status == "failed"
    assert result.metrics == {}
    if mode == "timeout":
        assert "timed out" in result.error
    else:
        assert "exited 7" in result.error
    pid = int((output / "pid").read_text())
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_missing_pyrosetta_dependency_is_an_explicit_worker_result(tmp_path, monkeypatch):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "pyrosetta.py").write_text('raise ModuleNotFoundError("PyRosetta test dependency unavailable")\n')
    monkeypatch.setenv("PYTHONPATH", str(modules))
    result = analysis._analyze_one(
        "complex.pdb", {"A": "AAA", "B": "AAA"}, "B_A", analysis.PyRosettaConfig(enabled=True), tmp_path / "output"
    )
    assert result.status == "failed"
    assert "PyRosetta test dependency unavailable" in result.error
    assert result.metrics == {}
    assert Path(tmp_path / "output/analysis.json").is_file()

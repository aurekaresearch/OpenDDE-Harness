const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const SERVER_PATH = path.resolve(__dirname, '..', 'server.js');

function freePort() {
  return new Promise((resolve, reject) => {
    const server = http.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
}

async function waitForServer(port, child) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (child.exitCode != null) throw new Error(`viewer exited with ${child.exitCode}`);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/api/health`);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error('viewer did not start');
}

test('serves the OpenDDE light purple tracing dashboard palette', async (t) => {
  const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'opendde-dashboard-palette-'));
  const port = await freePort();
  const child = spawn(process.execPath, [SERVER_PATH], {
    env: { ...process.env, TRACING_STATE_DIR: stateDir, TRACE_UI_PORT: String(port) },
    stdio: 'ignore'
  });
  t.after(() => {
    child.kill('SIGTERM');
    fs.rmSync(stateDir, { recursive: true, force: true });
  });
  await waitForServer(port, child);

  const response = await fetch(`http://127.0.0.1:${port}/app.css`);
  assert.equal(response.status, 200);
  const css = await response.text();

  const viewerResponse = await fetch(`http://127.0.0.1:${port}/vendor/3Dmol-min.js`);
  assert.equal(viewerResponse.status, 200);
  assert.match(await viewerResponse.text(), /3Dmol/);

  assert.match(css, /color-scheme: light;/);
  assert.match(css, /--accent: #6c43c9;/);
  assert.match(css, /--accent-bg: rgba\(108, 67, 201, 0\.1\);/);
  assert.match(css, /--accent-border: rgba\(108, 67, 201, 0\.36\);/);
  assert.match(css, /--purple: #8f6be7;/);
  assert.match(css, /--focus-ring: 0 0 0 3px rgba\(108, 67, 201, 0\.18\);/);
});

function writeFixture(stateDir, cycleArtifactPath) {
  const logs = path.join(stateDir, 'logs');
  fs.mkdirSync(logs, { recursive: true });
  const run = {
    schemaVersion: 'audit.span.v1',
    traceId: 'trace-task-1',
    spanId: 'run-task-1',
    parentSpanId: null,
    name: 'protein_design.run',
    startTime: '2026-08-18T00:00:00Z',
    endTime: '2026-08-18T00:01:00Z',
    status: { code: 'OK', message: '' },
    attributes: {
      'span.type': 'protein_design',
      'session.id': 'session-task-1',
      'protein_design.task_id': 'task-1',
      'protein_design.target': 'CRLF2',
      'protein_design.status': 'running',
      'protein_design.cycle': 1,
      'protein_design.total_cycles': 10,
      'protein_design.objective_key': 'loss',
      'protein_design.minimize': true
    }
  };
  const cycle = {
    ...run,
    spanId: 'cycle-task-1-0',
    parentSpanId: 'run-task-1',
    name: 'protein_design.cycle',
    attributes: {
      ...run.attributes,
      'protein_design.cycle_index': 0,
      'protein_design.cycle.artifact_path': cycleArtifactPath
    }
  };
  fs.writeFileSync(
    path.join(logs, 'audit-spans.log'),
    `${JSON.stringify(run)}\n${JSON.stringify(cycle)}\n`,
    'utf8'
  );
  const taskRoot = path.join(stateDir, 'protein-design-tasks');
  const taskDir = path.join(taskRoot, 'task-1');
  fs.mkdirSync(taskDir, { recursive: true });
  fs.writeFileSync(path.join(taskDir, 'workflow.json'), JSON.stringify({cdr_region_groups: {D: [[1, 2], [4, 5], [8, 9]]}}), 'utf8');
  return taskRoot;
}

test('serves projected OpenDDE Harness protein-design runs without changing the traces API', async (t) => {
  const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'opendde-protein-design-api-'));
  const artifacts = path.join(stateDir, 'logs', 'audit-artifacts');
  fs.mkdirSync(artifacts, { recursive: true });
  const cycleArtifactPath = path.join(artifacts, 'cycle.json');
  fs.writeFileSync(
    cycleArtifactPath,
    JSON.stringify({
      schema_version: 'protein_design.cycle.v1',
      task_id: 'task-1',
      target: 'CRLF2',
      cycle: 0,
      objective_key: 'loss',
      minimize: true,
      candidates: [{ candidate_id: 'candidate-1', sequence: 'ACDE', objective: 0.4, metrics: {} }],
      cycle_best_candidate_id: 'candidate-1',
      global_best_candidate_id: 'candidate-1'
    }),
    'utf8'
  );
  const taskRoot = writeFixture(stateDir, cycleArtifactPath);
  const port = await freePort();
  const child = spawn(process.execPath, [SERVER_PATH], {
    env: {
      ...process.env,
      TRACING_STATE_DIR: stateDir,
      TRACE_UI_PORT: String(port),
      OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT: taskRoot
    },
    stdio: 'ignore'
  });
  t.after(() => {
    child.kill('SIGTERM');
    fs.rmSync(stateDir, { recursive: true, force: true });
  });
  await waitForServer(port, child);

  const collection = await (await fetch(`http://127.0.0.1:${port}/api/protein-design`)).json();
  assert.equal(collection.selectedRunId, 'task-1');
  assert.equal(collection.runs[0].target, 'CRLF2');
  assert.equal(Object.hasOwn(collection, 'run'), false);

  const selected = await (
    await fetch(`http://127.0.0.1:${port}/api/protein-design?run_id=task-1`)
  ).json();
  assert.equal(selected.run.taskId, 'task-1');
  assert.deepEqual(selected.run.cdrRegionGroups, {D: [[1, 2], [4, 5], [8, 9]]});
  assert.equal(selected.run.cycles[0].candidates[0].candidateId, 'candidate-1');
  const traces = await fetch(`http://127.0.0.1:${port}/api/data`);
  assert.equal(traces.status, 200);
});

test('does not follow a protein-design artifact symlink outside the tracing root', async (t) => {
  const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'opendde-protein-design-safe-'));
  const artifacts = path.join(stateDir, 'logs', 'audit-artifacts');
  fs.mkdirSync(artifacts, { recursive: true });
  const outside = path.join(stateDir, 'outside.json');
  fs.writeFileSync(
    outside,
    JSON.stringify({ schema_version: 'protein_design.cycle.v1', cycle: 0, candidates: [] }),
    'utf8'
  );
  const linked = path.join(artifacts, 'linked.json');
  fs.symlinkSync(outside, linked);
  const taskRoot = writeFixture(stateDir, linked);
  const port = await freePort();
  const child = spawn(process.execPath, [SERVER_PATH], {
    env: {
      ...process.env,
      TRACING_STATE_DIR: stateDir,
      TRACE_UI_PORT: String(port),
      OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT: taskRoot
    },
    stdio: 'ignore'
  });
  t.after(() => {
    child.kill('SIGTERM');
    fs.rmSync(stateDir, { recursive: true, force: true });
  });
  await waitForServer(port, child);

  const projected = await (
    await fetch(`http://127.0.0.1:${port}/api/protein-design?run_id=task-1`)
  ).json();
  assert.deepEqual(projected.run.cycles, []);
});

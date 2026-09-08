const test = require('node:test');
const assert = require('node:assert/strict');

const {
  flattenFiniteMetrics,
  projectProteinDesignRuns
} = require('../protein-design-store');

function span({
  spanId,
  name,
  taskId,
  status = 'running',
  startTime = '2026-08-18T00:00:00.000Z',
  endTime = '2026-08-18T00:01:00.000Z',
  parentSpanId = null,
  attributes = {}
}) {
  return {
    schemaVersion: 'audit.span.v1',
    traceId: `trace-${taskId}`,
    spanId,
    parentSpanId,
    name,
    startTime,
    endTime,
    status: { code: 'OK', message: '' },
    attributes: {
      'span.type': 'protein_design',
      'session.id': `session-${taskId}`,
      'protein_design.task_id': taskId,
      'protein_design.status': status,
      ...attributes
    }
  };
}

function runSpan(taskId, options = {}) {
  return span({
    spanId: options.spanId || `run-${taskId}`,
    name: 'protein_design.run',
    taskId,
    status: options.status || 'running',
    startTime: options.startTime,
    endTime: options.endTime,
    attributes: {
      'protein_design.target': options.target || 'CRLF2',
      'protein_design.phase': options.phase || '',
      'protein_design.cycle': options.cycle || 0,
      'protein_design.total_cycles': options.totalCycles || 100,
      'protein_design.objective_key': options.objectiveKey || 'loss',
      'protein_design.minimize': options.minimize ?? true,
      'protein_design.best_candidate_id': options.bestCandidateId || null,
      'protein_design.best_objective': options.bestObjective ?? null,
      'protein_design.compute_url': options.computeUrl || null,
      'protein_design.compute_worker_id': options.computeWorkerId || null,
      'protein_design.final_selection.artifact_path': options.finalSelectionArtifactPath || null
    }
  });
}

test('deduplicates checkpoints and selects newest active run before recent completed runs', () => {
  const spans = [
    runSpan('active-old', { spanId: 'same-run', cycle: 1, startTime: '2026-08-18T01:00:00Z' }),
    runSpan('completed-newer', {
      status: 'completed',
      startTime: '2026-08-18T03:00:00Z',
      endTime: '2026-08-18T04:00:00Z'
    }),
    runSpan('active-latest', { startTime: '2026-08-18T02:00:00Z' }),
    runSpan('active-old', { spanId: 'same-run', cycle: 7, startTime: '2026-08-18T01:00:00Z' })
  ];

  const projected = projectProteinDesignRuns({ spans, readArtifact: () => null });

  assert.deepEqual(projected.runs.map((run) => run.taskId), [
    'active-latest',
    'active-old',
    'completed-newer'
  ]);
  assert.equal(projected.selectedRunId, 'active-latest');
  assert.equal(projected.runs[1].cycle, 7);
});

test('filters trace-only placeholder runs against registered detached tasks', () => {
  const projected = projectProteinDesignRuns({
    spans: [runSpan('task'), runSpan('real-task')],
    registeredTaskIds: new Set(['real-task']),
    readArtifact: () => null
  });

  assert.deepEqual(projected.runs.map((run) => run.taskId), ['real-task']);
  assert.equal(projected.selectedRunId, 'real-task');
});

test('returns no runs when no detached protein-design tasks are registered', () => {
  const projected = projectProteinDesignRuns({
    spans: [runSpan('task')],
    registeredTaskIds: new Set(),
    readArtifact: () => null
  });

  assert.deepEqual(projected, { runs: [], selectedRunId: null, run: null });
});

test('projects the task-level compute binding', () => {
  const projected = projectProteinDesignRuns({
    spans: [runSpan('remote', {
      computeUrl: 'http://zkln-10:8080',
      computeWorkerId: 'zkln-10'
    })],
    readArtifact: () => null
  });

  assert.equal(projected.run.computeUrl, 'http://zkln-10:8080');
  assert.equal(projected.run.computeWorkerId, 'zkln-10');
});

test('projects dynamic cycle and global-best metric series without conflating candidates', () => {
  const cycleArtifact = {
    schema_version: 'protein_design.cycle.v1',
    task_id: 'task-1',
    target: 'CRLF2',
    cycle: 0,
    objective_key: 'loss',
    minimize: true,
    candidates: [
      {
        candidate_id: 'cycle-best',
        sequence: 'ACDE',
        objective: 0.4,
        metrics: {
          iptm: 0.7,
          confidence: { plddt: 82.5 },
          gate_passed: 1.0,
          cdr3_gate_passed: 1.0,
          cdr_contact_fraction_gate_passed: 1.0,
          invalid: Infinity
        },
        parent_id: 'missing-parent',
        skill_id: 'cdr-point-mutation',
        metadata: {
          gate_passed: true,
          gate_evidence: {
            cdr_contact_fraction: 0.84,
            cdr_total_contacts: 11,
            framework_total_contacts: 2,
            framework_contact_residue_ids: [
              { chain: 'D', residue_id: 19, distance_to_antigen_a: 4.2 }
            ]
          },
          chains: { D: 'ACDE' },
          fold: { sequences: { A: 'TARGET', D: 'ACDE' } }
        },
        structure_artifact_path: '/traces/artifacts/structure.json'
      },
      {
        candidate_id: 'global-best',
        sequence: 'FGHI',
        objective: 0.3,
        metrics: { iptm: 0.6, confidence: { plddt: 91.0 } },
        parent_id: null,
        skill_id: 'full-redesign',
        status: 'failed',
        structure_artifact_path: null
      }
    ],
    cycle_best_candidate_id: 'cycle-best',
    global_best_candidate_id: 'global-best',
    population_candidate_ids: ['cycle-best'],
    admitted_candidate_ids: ['cycle-best'],
    population_actions: { 'cycle-best': 'inserted', 'global-best': 'gate_rejected' }
  };
  const finalSelection = {
    post_filter_enabled: true,
    post_filter_executed: true,
    post_filter_top_k: 1,
    mode: 'agent',
    strategy_summary: 'Prefer strong CDR contact without framework contact.',
    refolded_candidate_count: 2,
    eligible_candidate_count: 1,
    selected_candidate_ids: ['cycle-best'],
    structure_artifacts: { 'cycle-best': '/traces/artifacts/refolded-structure.json' },
    target_chain_ids: ['A'],
    binder_chain_ids: ['D'],
    candidates: [{
      candidate_id: 'cycle-best',
      sequence: 'ACDE',
      metrics: { iptm: 0.81, plddt: 0.88, cdr_contact_fraction: 0.84, gate_passed: 1 }
    }],
    decisions: [
      {
        candidate_id: 'cycle-best', rank: 1, score: 0.91, objective: 0.4,
        pass_filter: true, hard_eligible: true, rationale: 'Best geometry.',
        strengths: ['CDR contact'], risks: []
      }
    ]
  };
  const spans = [
    runSpan('task-1', {
      bestCandidateId: 'global-best',
      bestObjective: 0.3,
      finalSelectionArtifactPath: '/traces/artifacts/final-selection.json'
    }),
    span({
      spanId: 'cycle-0',
      name: 'protein_design.cycle',
      taskId: 'task-1',
      parentSpanId: 'run-task-1',
      attributes: {
        'protein_design.cycle_index': 0,
        'protein_design.cycle.artifact_path': '/traces/artifacts/cycle.json'
      }
    })
  ];
  const artifacts = new Map([
    ['/traces/artifacts/cycle.json', cycleArtifact],
    ['/traces/artifacts/final-selection.json', finalSelection]
  ]);

  const projected = projectProteinDesignRuns({
    spans,
    selectedRunId: 'task-1',
    readArtifact: (artifactPath) => artifacts.get(artifactPath) || null
  });

  assert.deepEqual(projected.run.metricNames, ['confidence.plddt', 'iptm', 'loss']);
  assert.deepEqual(projected.run.series.loss.cycleBest, [
    { cycle: 0, candidateId: 'cycle-best', value: 0.4 }
  ]);
  assert.deepEqual(projected.run.series.loss.globalBest, [
    { cycle: 0, candidateId: 'global-best', value: 0.3 }
  ]);
  assert.equal(projected.run.structures[0].candidateId, 'cycle-best');
  assert.deepEqual(projected.run.structures[0].targetChainIds, ['A']);
  assert.deepEqual(projected.run.structures[0].binderChainIds, ['D']);
  assert.equal(projected.run.structures[0].gatePassed, true);
  assert.equal(projected.run.structures[0].gateMetrics.cdr_contact_fraction, 0.84);
  assert.equal(projected.run.structures[0].gateMetrics.gate_passed, 1.0);
  assert.equal(projected.run.structures[0].gateMetrics.cdr3_gate_passed, 1.0);
  assert.equal(
    projected.run.structures[0].gateMetrics.cdr_contact_fraction_gate_passed,
    1.0
  );
  assert.equal(projected.run.structures[0].inPopulation, true);
  assert.deepEqual(projected.run.populationCandidateIds, ['cycle-best']);
  assert.equal(projected.run.population[0].candidateId, 'cycle-best');
  assert.equal(projected.run.postFilter.status, 'completed');
  assert.deepEqual(projected.run.postFilter.selectedCandidateIds, ['cycle-best']);
  assert.equal(
    projected.run.postFilter.decisions[0].structureArtifactPath,
    '/traces/artifacts/refolded-structure.json'
  );
  assert.deepEqual(projected.run.postFilter.decisions[0].targetChainIds, ['A']);
  assert.deepEqual(projected.run.postFilter.decisions[0].binderChainIds, ['D']);
  assert.equal(projected.run.postFilter.decisions[0].sequence, 'ACDE');
  assert.equal(projected.run.postFilter.decisions[0].metrics.iptm, 0.81);
  assert.equal(projected.run.postFilter.decisions[0].metrics.plddt, 0.88);
  assert.equal(projected.run.postFilter.decisions[0].metrics.cdr_contact_fraction, 0.84);
  assert.equal(projected.run.postFilter.decisions[0].metrics.gate_passed, undefined);
  assert.equal(projected.run.failureCount, 1);
  const candidateNode = projected.run.tree.nodes.find((node) => node.id === 'candidate:cycle-best');
  assert.equal(candidateNode.parentId, 'candidate:missing-parent');
  assert.equal(
    projected.run.tree.nodes.find((node) => node.id === 'candidate:missing-parent').kind,
    'initial'
  );
});

test('falls back to scored cycle best and preserves an initial-parent lineage', () => {
  const artifact = {
    schema_version: 'protein_design.cycle.v1',
    task_id: 'rejected-run',
    cycle: 0,
    candidates: [
      {
        candidate_id: 'proposal-a',
        objective: 3.4,
        metrics: { iptm: 0.72, loss: 3.4 },
        parent_id: 'initial-parent',
        status: 'scored'
      },
      {
        candidate_id: 'proposal-b',
        objective: 4.1,
        metrics: { iptm: 0.51, loss: 4.1 },
        parent_id: 'initial-parent',
        status: 'scored'
      }
    ],
    cycle_best_candidate_id: null,
    global_best_candidate_id: 'initial-parent'
  };
  const spans = [
    runSpan('rejected-run', {
      bestCandidateId: 'initial-parent',
      bestObjective: 4.8
    }),
    span({
      spanId: 'rejected-cycle',
      name: 'protein_design.cycle',
      taskId: 'rejected-run',
      attributes: {
        'protein_design.cycle_index': 0,
        'protein_design.global_best_objective': 4.8,
        'protein_design.cycle.artifact_path': '/rejected-cycle.json'
      }
    })
  ];

  const projected = projectProteinDesignRuns({
    spans,
    readArtifact: (path) => path === '/rejected-cycle.json' ? artifact : null
  });

  assert.deepEqual(projected.run.series.loss.cycleBest, [
    { cycle: 0, candidateId: 'proposal-a', value: 3.4 }
  ]);
  assert.deepEqual(projected.run.series.loss.globalBest, [
    { cycle: 0, candidateId: 'initial-parent', value: 4.8 }
  ]);
  assert.deepEqual(projected.run.series.iptm.cycleBest, [
    { cycle: 0, candidateId: 'proposal-a', value: 0.72 }
  ]);
  const initialNode = projected.run.tree.nodes.find(
    (node) => node.id === 'candidate:initial-parent'
  );
  const proposalNode = projected.run.tree.nodes.find(
    (node) => node.id === 'candidate:proposal-a'
  );
  assert.equal(initialNode.kind, 'initial');
  assert.equal(proposalNode.parentId, 'candidate:initial-parent');
  assert.equal(projected.run.tree.nodes.some((node) => node.kind === 'cycle'), false);
});

test('flattens only finite numeric metrics and rejects booleans arrays and strings', () => {
  assert.deepEqual(
    flattenFiniteMetrics({
      iptm: 0.7,
      nested: { plddt: 92, enabled: true },
      values: [1, 2],
      label: 'high',
      nan: Number.NaN,
      infinity: Number.POSITIVE_INFINITY
    }),
    { iptm: 0.7, 'nested.plddt': 92 }
  );
});

test('ignores unsupported cycle artifacts and honors an explicit valid run selection', () => {
  const spans = [
    runSpan('active', { startTime: '2026-08-18T02:00:00Z' }),
    runSpan('selected', { status: 'completed', startTime: '2026-08-18T01:00:00Z' }),
    span({
      spanId: 'selected-cycle',
      name: 'protein_design.cycle',
      taskId: 'selected',
      attributes: {
        'protein_design.cycle_index': 0,
        'protein_design.cycle.artifact_path': '/bad-schema.json'
      }
    })
  ];

  const projected = projectProteinDesignRuns({
    spans,
    selectedRunId: 'selected',
    readArtifact: () => ({ schema_version: 'protein_design.cycle.v9', candidates: [] })
  });

  assert.equal(projected.selectedRunId, 'selected');
  assert.deepEqual(projected.run.cycles, []);
  assert.deepEqual(projected.run.metricNames, []);
});

test('projects full structured progress artifacts for the selected run', () => {
  const spans = [
    runSpan('task-live'),
    span({
      spanId: 'progress-1',
      name: 'protein_design.progress',
      taskId: 'task-live',
      startTime: '2026-08-18T00:00:10Z',
      attributes: {
        'protein_design.progress.artifact_path': '/progress/1.json'
      }
    })
  ];
  const event = {
    event_id: 'event-1',
    task_id: 'task-live',
    timestamp: '2026-08-18T00:00:10Z',
    event_type: 'agent',
    status: 'completed',
    cycle: 3,
    phase: 'design',
    actor: 'design',
    skill: 'cdr-point-mutation',
    duration_ms: 1250,
    candidate_count: 8,
    input_payload: { prompt: 'full prompt' },
    output_payload: { skill_id: 'cdr-point-mutation' }
  };
  const projected = projectProteinDesignRuns({
    spans,
    readArtifact: (path) => path === '/progress/1.json' ? event : null
  });
  assert.deepEqual(projected.run.events, [event]);
});

test('keeps one completed activity with IO and hides empty orchestration events', () => {
  const events = [
    {
      event_id: 'phase-start', task_id: 'task-live', timestamp: '2026-08-18T00:00:01Z',
      event_type: 'agent', status: 'started', cycle: 3, phase: 'design', actor: 'design-agent'
    },
    {
      event_id: 'agent-start', task_id: 'task-live', timestamp: '2026-08-18T00:00:02Z',
      event_type: 'agent', status: 'started', cycle: 3, phase: 'design', actor: 'design',
      input_payload: { prompt: 'design prompt' }
    },
    {
      event_id: 'agent-complete', task_id: 'task-live', timestamp: '2026-08-18T00:00:03Z',
      event_type: 'agent', status: 'completed', cycle: 3, phase: 'design', actor: 'design',
      input_payload: { prompt: 'design prompt' }, output_payload: { selected_skill_id: 'cdr-point-mutation' }
    },
    {
      event_id: 'phase-complete', task_id: 'task-live', timestamp: '2026-08-18T00:00:04Z',
      event_type: 'agent', status: 'completed', cycle: 3, phase: 'design', actor: 'design-agent'
    },
    {
      event_id: 'quality-start', task_id: 'task-live', timestamp: '2026-08-18T00:00:05Z',
      event_type: 'agent', status: 'started', cycle: 3, phase: 'quality', actor: 'design-agent'
    },
    {
      event_id: 'quality-complete', task_id: 'task-live', timestamp: '2026-08-18T00:00:06Z',
      event_type: 'agent', status: 'completed', cycle: 3, phase: 'quality', actor: 'design-agent'
    }
  ];
  const spans = [runSpan('task-live')];
  const artifacts = new Map();
  events.forEach((event, index) => {
    const path = `/progress/${index}.json`;
    artifacts.set(path, event);
    spans.push(span({
      spanId: `progress-${index}`,
      name: 'protein_design.progress',
      taskId: 'task-live',
      startTime: event.timestamp,
      attributes: { 'protein_design.progress.artifact_path': path }
    }));
  });

  const projected = projectProteinDesignRuns({
    spans,
    readArtifact: (path) => artifacts.get(path) || null
  });

  assert.deepEqual(projected.run.events.map((event) => event.event_id), ['agent-complete']);
  assert.equal(projected.run.eventCount, 1);
  assert.equal(projected.run.progressEventCount, 6);
});

test('keeps an unfinished agent activity visible while it is running', () => {
  const event = {
    event_id: 'agent-start', task_id: 'task-live', timestamp: '2026-08-18T00:00:02Z',
    event_type: 'agent', status: 'started', cycle: 3, phase: 'design', actor: 'design',
    input_payload: { prompt: 'design prompt' }
  };
  const spans = [
    runSpan('task-live'),
    span({
      spanId: 'progress-start', name: 'protein_design.progress', taskId: 'task-live',
      startTime: event.timestamp,
      attributes: { 'protein_design.progress.artifact_path': '/progress/start.json' }
    })
  ];

  const projected = projectProteinDesignRuns({
    spans,
    readArtifact: (path) => path === '/progress/start.json' ? event : null
  });

  assert.deepEqual(projected.run.events, [event]);
});


test('post-filter failures are failed, never fallback, and obsolete scores are ignored', () => {
  for (const failure of [
    { mode: 'failed', post_filter_error: 'agent unavailable' },
    { mode: 'failed', post_refold_error: 'refold unavailable' },
    { mode: 'agent', post_filter_error: 'invalid ranking' }
  ]) {
    const artifact = {
      post_filter_enabled: true,
      post_filter_executed: true,
      refolded_candidate_count: 0,
      eligible_candidate_count: 0,
      selected_candidate_ids: [],
      candidates: [{ candidate_id: 'c1', sequence: 'ACDE' }],
      metric_weights: [{ metric: 'objective', weight: 1 }],
      decisions: [{ candidate_id: 'c1', rank: 1, score: 0.9 }],
      ...failure
    };
    const projected = projectProteinDesignRuns({
      spans: [runSpan('task-1', { finalSelectionArtifactPath: '/final.json' })],
      selectedRunId: 'task-1',
      readArtifact: artifactPath => artifactPath === '/final.json' ? artifact : null
    });
    const result = projected.run.postFilter;
    assert.equal(result.status, 'failed');
    assert.equal(result.refoldedCandidateCount, 0);
    assert.deepEqual(result.selectedCandidateIds, []);
    assert.equal(Object.hasOwn(result, 'metricWeights'), false);
    assert.equal(Object.hasOwn(result.decisions[0], 'score'), false);
    assert.equal(result.decisions[0].rank, 1);
  }
});

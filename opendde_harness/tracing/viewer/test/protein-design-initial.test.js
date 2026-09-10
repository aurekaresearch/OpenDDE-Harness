const test = require('node:test')
const assert = require('node:assert/strict')
const { projectProteinDesignRuns } = require('../protein-design-store')

function span({ spanId, name, taskId, attributes }) {
  return {
    spanId,
    name,
    traceId: 'trace-initial',
    attributes: {
      'protein_design.task_id': taskId,
      ...attributes
    }
  }
}

function runSpan(taskId, options) {
  return span({
    spanId: 'run-initial',
    name: 'protein_design.run',
    taskId,
    attributes: {
      'protein_design.status': 'running',
      'protein_design.cycle': options.cycle,
      'protein_design.phase': options.phase,
      'protein_design.best_candidate_id': options.bestCandidateId,
      'protein_design.best_objective': options.bestObjective
    }
  })
}

test('shows initial structures and scores while the first design cycle is retrying', () => {
  const artifact = {
    schema_version: 'protein_design.cycle.v1',
    task_id: 'initial-task',
    cycle: -1,
    candidates: [
      {
        candidate_id: 'seed',
        sequence: 'ACDE',
        objective: 1.8,
        metrics: { loss: 1.8, iptm: 0.34 },
        status: 'scored',
        structure_artifact_path: '/traces/seed.json'
      }
    ],
    cycle_best_candidate_id: 'seed',
    global_best_candidate_id: 'seed',
    population_candidate_ids: ['seed'],
    admitted_candidate_ids: ['seed']
  }
  const projected = projectProteinDesignRuns({
    spans: [
      runSpan('initial-task', { cycle: 0, phase: 'cycle_retry', bestCandidateId: 'seed', bestObjective: 1.8 }),
      span({
        spanId: 'initial-fold',
        name: 'protein_design.cycle',
        taskId: 'initial-task',
        attributes: {
          'protein_design.cycle_index': -1,
          'protein_design.cycle.artifact_path': '/traces/initial.json'
        }
      })
    ],
    readArtifact: path => (path === '/traces/initial.json' ? artifact : null)
  })
  assert.equal(projected.run.cycle, 0)
  assert.equal(projected.run.candidateCount, 1)
  assert.equal(projected.run.cycles[0].cycle, -1)
  assert.equal(projected.run.structures[0].artifactPath, '/traces/seed.json')
  assert.deepEqual(projected.run.populationCandidateIds, ['seed'])
  assert.equal(projected.run.series.iptm.cycleBest[0].value, 0.34)
})

test('fold checkpoints stay visible and completion replaces the same round', () => {
  const artifacts = {
    early: {
      schema_version: 'protein_design.cycle.v1',
      cycle: 1,
      candidates: [
        {
          candidate_id: 'round1',
          objective: 2,
          metrics: { loss: 2 },
          status: 'scored',
          structure_artifact_path: '/round1.json'
        }
      ]
    }
  }
  artifacts.complete = { ...artifacts.early, admitted_candidate_ids: ['round1'], population_candidate_ids: ['round1'] }
  const checkpoint = path =>
    span({
      spanId: 'round1',
      name: 'protein_design.cycle',
      taskId: 'task',
      attributes: { 'protein_design.cycle_index': 1, 'protein_design.cycle.artifact_path': path }
    })
  const root = runSpan('task', { cycle: 1, phase: 'quality' })
  for (const records of [[checkpoint('early')], [checkpoint('early'), checkpoint('complete')]]) {
    const { run } = projectProteinDesignRuns({ spans: [root, ...records], readArtifact: path => artifacts[path] })
    assert.equal(run.candidateCount, 1)
    assert.equal(run.cycles.length, 1)
    assert.equal(run.structures.length, 1)
    assert.equal(run.structures[0].artifactPath, '/round1.json')
    assert.equal(run.structures[0].admitted, records.length === 2)
  }
})

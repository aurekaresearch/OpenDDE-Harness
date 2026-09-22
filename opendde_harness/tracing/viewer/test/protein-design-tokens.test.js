const test = require('node:test')
const assert = require('node:assert/strict')
const { projectProteinDesignRuns } = require('../protein-design-store')
const { renderActivity } = require('../ui/protein-design')

function project(usages, status = 'completed', withOverlap = false) {
  const artifacts = {}
  const spans = [
    { spanId: 'run', traceId: 't', name: 'protein_design.run', attributes: { 'protein_design.task_id': 'task' } }
  ]
  for (const [index, state] of ['started', status].entries()) {
    const path = `progress-${index}`
    artifacts[path] = {
      task_id: 'task',
      event_type: 'agent',
      actor: 'design',
      phase: 'design',
      cycle: 1,
      status: state,
      timestamp: `2026-09-21T00:00:${index ? '20' : '00'}Z`,
      input_payload: { prompt: 'test' },
      duration_ms: index ? 20000 : null
    }
    spans.push({
      spanId: path,
      traceId: 't',
      name: 'protein_design.progress',
      attributes: {
        'protein_design.task_id': 'task',
        'protein_design.progress.artifact_path': path
      }
    })
  }
  usages.forEach((usage, index) => {
    const path = `output-${index}`
    artifacts[path] = { usage }
    spans.push({
      spanId: path,
      traceId: 't',
      name: 'llm.call',
      startTime: '2026-09-21T00:00:01Z',
      endTime: '2026-09-21T00:00:19Z',
      attributes: { 'llm.output.artifact_path': path }
    })
  })
  // Checkpoints must not double count; another trace must not contribute.
  if (usages.length) spans.push(spans.at(-1), { ...spans.at(-1), spanId: 'other', traceId: 'unrelated' })
  if (withOverlap) {
    artifacts['other-input'] = { prompt: 'another speculative design' }
    spans.push({
      spanId: 'overlap',
      traceId: 't',
      name: 'llm.call',
      startTime: '2026-09-21T00:00:02Z',
      endTime: '2026-09-21T00:00:03Z',
      attributes: { 'llm.input.artifact_path': 'other-input', 'llm.usage.input_tokens': 99999 }
    })
  }
  return projectProteinDesignRuns({ spans, readArtifact: path => artifacts[path] }).run.events[0]
}

const known = {
  prompt_tokens: 100,
  completion_tokens: 50,
  cache_read_input_tokens: 800,
  cache_creation_input_tokens: 100
}

test('aggregates calls, including retries in a failed phase, without double counting', () => {
  const event = project([known, known], 'failed')
  assert.equal(event.token_usage.calls, 2)
  assert.deepEqual(event.token_usage.fields.input, { value: 2000, reported: 2 })
  assert.equal(event.token_usage.fields.output.value, 100)
  assert.equal(event.token_usage.fields.cached.value, 1600)
  assert.equal(event.token_usage.fields.written.value, 200)
  assert.equal(event.token_usage.hitRate, 0.8)
  const html = renderActivity([event])
  assert.match(html, /Input \(total\)/)
  assert.match(html, /2,000/)
  assert.match(html, /80.0%/)
})

test('missing usage stays unknown, and partial totals are labelled', () => {
  const event = project([known, {}])
  assert.equal(event.token_usage.fields.input.reported, 1)
  assert.equal(event.token_usage.hitRate, null)
  assert.match(renderActivity([event]), /≥ 1,000/)
  assert.match(renderActivity([event]), /1\/2 reported/)
  assert.match(renderActivity([project([{}])]), /Unknown/)
})

test('explicit zero cache is different from a missing cache field', () => {
  const zero = project([{ ...known, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }])
  assert.equal(zero.token_usage.fields.cached.value, 0)
  assert.equal(zero.token_usage.hitRate, 0)
  const missing = project([{ prompt_tokens: 100, completion_tokens: 50 }])
  assert.equal(missing.token_usage.fields.cached.value, null)
  assert.equal(missing.token_usage.fields.output.value, 50)
  assert.equal(missing.token_usage.fields.input.value, null)
})

test('zero-filled missing statistics and no calls do not imply a hit rate of zero', () => {
  const empty = project([
    { prompt_tokens: 0, completion_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 }
  ])
  assert.equal(empty.token_usage.fields.cached.value, null)
  assert.equal(project([]).token_usage.hitRate, null)
})

test('overlapping speculative calls with a different prompt are not attributed to the agent', () => {
  assert.equal(project([known], 'completed', true).token_usage.calls, 1)
})

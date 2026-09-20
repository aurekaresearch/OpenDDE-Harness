const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const shell = require('../ui/shell')
const {
  PROPERTY_DEFINITIONS,
  metricLabel,
  metricDescription,
  refreshTableFilterRange,
  tableFilterSliderPosition,
  updateTableFilterRange,
  revealCandidateRow,
  alignStructureArtifacts,
  candidateCdrRegions,
  candidateGroups,
  chartCandidates,
  compactSequence,
  captureDashboardViewState,
  defaultSelectedKeys,
  minIpaColor,
  parallelGeometry,
  pdbAlphaCarbons,
  propertyDefinitions,
  propertyValue,
  propertyTicks,
  renderParallelCoordinates,
  renderProteinDesignDashboard,
  resultRun,
  restoreDashboardViewState,
  selectRunId,
  shortTaskId
} = require('../ui/protein-design-candidates')

test('post-filter reuses candidate rows and refolded metrics without mutating design results', () => {
  const original = {
    candidateId: 'a',
    cycle: 3,
    sequence: 'AAA',
    metrics: { iptm: 0.1, loss: 5 },
    structureArtifactPath: '/old.json'
  }
  const run = {
    taskId: 'run',
    cycles: [{ cycle: 3, candidates: [original] }],
    structures: [{ candidateId: 'a', cycle: 3, artifactPath: '/old.json' }],
    postFilter: {
      decisions: [
        {
          candidateId: 'a',
          rank: 1,
          passFilter: true,
          rationale: 'Best geometry',
          metrics: { iptm: 0.9, plddt: 0.8 },
          structureArtifactPath: '/refold.json',
          targetChainIds: ['A'],
          binderChainIds: ['B']
        },
        {
          candidateId: 'b',
          rank: 2,
          passFilter: false,
          hardEligible: false,
          sequence: 'BBB',
          metrics: {},
          structureArtifactPath: null
        }
      ]
    }
  }
  assert.equal(resultRun(run, 'design'), run)
  const projected = resultRun(run, 'post-filter')
  const rows = candidateGroups(projected)
  assert.equal(rows.length, 2)
  assert.equal(rows[0].candidates[0].sequence, 'AAA')
  assert.deepEqual(rows[0].candidates[0].metrics, { iptm: 0.9, plddt: 0.8 })
  assert.equal(projected.structures[0].artifactPath, '/refold.json')
  assert.deepEqual(projected.structures[0].binderChainIds, ['B'])
  assert.equal(original.metrics.iptm, 0.1)
  const html = renderProteinDesignDashboard({ run: projected })
  assert.match(html, /protein-candidate-table/)
  assert.match(html, /protein-inspector-panel/)
  assert.match(html, /#1 · Selected/)
  assert.match(html, /Best geometry/)
  assert.match(html, /Hard rejected/)
  assert.doesNotMatch(html, /protein-filter-structure-panel|View other sequences/)
  assert.match(html, /value="metric:plddt">pLDDT/)
  assert.match(html, /data-table-metrics=".*&quot;plddt&quot;:0.8/)
})

test('post-filter does not reuse design candidates when decisions are absent', () => {
  const run = { cycles: [{ cycle: 1, candidates: [{ candidateId: 'design-only' }] }] }
  const projected = resultRun(run, 'post-filter')
  assert.deepEqual(candidateGroups(projected), [])
  assert.deepEqual(projected.structures, [])
})

test('post-filter reads final contact and loss evidence without borrowing design evidence', () => {
  const oldMetadata = {
    gate_evidence: { cdr_total_contacts: 999, framework_total_contacts: 999 },
    loss: { loss_components: { i_pae: 999 } }
  }
  const run = {
    cycles: [
      {
        cycle: 1,
        candidates: [
          { candidateId: 'ok', metadata: oldMetadata },
          { candidateId: 'failed', metadata: oldMetadata }
        ]
      }
    ],
    postFilter: {
      decisions: [
        {
          candidateId: 'ok',
          metrics: { iptm: 0.8, ranking_score: 0.7, loss: 1.2, min_ipae: 0.3 },
          metadata: {
            gate_evidence: { cdr_total_contacts: 66, framework_total_contacts: 0 },
            loss: { loss_components: { i_pae: 0.3 } }
          }
        },
        { candidateId: 'failed', metrics: {} }
      ]
    }
  }
  const projected = resultRun(run, 'post-filter')
  const [ok, failed] = candidateGroups(projected).map(group => group.candidates[0])
  const value = (candidate, key) =>
    propertyValue(
      candidate,
      PROPERTY_DEFINITIONS.find(item => item.key === key)
    )
  assert.equal(value(ok, 'contacts'), 66)
  assert.equal(value(ok, 'frame_contacts'), 0)
  assert.equal(value(ok, 'min_ipa'), 0.3)
  for (const key of ['contacts', 'frame_contacts', 'min_ipa']) assert.equal(value(failed, key), null)
  assert.equal(oldMetadata.gate_evidence.cdr_total_contacts, 999)
  const complete = { ...projected, cycles: [{ cycle: 1, candidates: [ok] }] }
  assert.doesNotMatch(renderProteinDesignDashboard({ run: complete }), /metrics incomplete/)
  assert.match(renderProteinDesignDashboard({ run: projected }), /metrics incomplete/)
})

test('metric labels use canonical scientific capitalization', () => {
  assert.equal(metricLabel('cdr3_gate_passed'), 'CDR3 gate passed')
  for (const [key, label] of Object.entries({
    iptm: 'ipTM',
    ptm: 'pTM',
    plddt: 'pLDDT',
    pae: 'PAE',
    min_ipae: 'Min ipAE',
    ipsae: 'ipSAE',
    ranking_score: 'Ranking score',
    'confidence.plddt': 'pLDDT'
  })) {
    assert.equal(metricLabel(key), label)
  }
})

test('Rosetta metrics are selectable and analysis status exposes escaped loss and failure details', () => {
  const metrics = { rosetta_interface_dg: -12.5, rosetta_interface_sc: 0.65 }
  const metadata = {
    pyrosetta: {
      status: 'success',
      elapsed_seconds: 2,
      provenance: { score_function: 'ref2015' },
      contact_residues: [
        { chain_id: 'B', residue_index: 9, amino_acid: 'W', bound_score_reu: -2, interface_dg_reu: -1 }
      ]
    },
    loss: {
      components: {
        rosetta_interface_dg: { raw: -12.5, direction: 'minimize', weight: 0.5, scale: 10, contribution: -0.625 }
      }
    }
  }
  const candidate = { candidateId: 'relaxed', sequence: 'AAA', cycle: 1, metrics, metadata }
  const run = { cycles: [{ cycle: 1, candidates: [candidate] }] }
  const html = renderProteinDesignDashboard({ run })
  assert.match(html, /PyRosetta: success/)
  assert.match(html, /FastRelax → InterfaceAnalyzer/)
  assert.match(html, /loss contribution -0.625/)
  assert.match(html, /B\[9\] W: bound -2, interface ΔG -1/)
  assert.match(html, /value="metric:rosetta_interface_dg">Interface ΔG \(REU\)/)
  assert.match(html, /&quot;rosetta_interface_dg&quot;:-12.5/)
  assert.equal(metricLabel('rosetta_interface_sasa'), 'Interface ΔSASA (Å²)')
  metadata.pyrosetta = { status: 'failed', error: '<script>bad</script>' }
  const failed = renderProteinDesignDashboard({ run })
  assert.match(failed, /PyRosetta: failed/)
  assert.match(failed, /&lt;script&gt;bad&lt;\/script&gt;/)
  assert.doesNotMatch(failed, /<script>bad/)

  run.postFilter = { decisions: [{ candidateId: 'relaxed', metrics: {} }] }
  const refold = resultRun(run, 'post-filter')
  assert.equal(candidateGroups(refold)[0].candidates[0].metadata.pyrosetta, null)
  assert.doesNotMatch(renderProteinDesignDashboard({ run: refold }), /PyRosetta: failed/)
})

test('loss breakdown shows stored coefficients and signed values with full precision', () => {
  const loss = {
    structure_loss: 2,
    esm2_pll: -1,
    esm2_weight: 0.1,
    esm2_contribution: 0.1,
    metric_loss: -3.2123456789,
    loss: -1.1123456789,
    components: {
      rosetta_interface_dg: {
        raw: -64,
        direction: 'minimize',
        weight: 0.5,
        reference: 0,
        scale: 10,
        normalized: -6.4,
        contribution: -3.2
      },
      rosetta_interface_sc: {
        raw: 0.5123456789,
        direction: 'maximize',
        weight: 1,
        reference: 0.5,
        scale: 1,
        normalized: 0.0123456789,
        contribution: -0.0123456789
      }
    }
  }
  const candidate = { candidateId: 'scored', metrics: { loss: -1.1123456789 }, metadata: { loss } }
  const projected = { objectiveKey: 'loss', minimize: true, cycles: [{ cycle: 1, candidates: [candidate] }] }
  const html = renderProteinDesignDashboard({ run: projected })
  assert.match(html, /<details class="protein-loss-details"><summary>Loss breakdown · minimize/)
  assert.match(html, /Original \/ base loss<\/strong> <span data-value="2.1"/)
  assert.match(html, /Metric subtotal<\/strong> <span data-value="-3.2123456789"/)
  assert.match(html, /Composite loss<\/strong> <span data-value="-1.1123456789"/)
  assert.match(html, /<th>Reference<\/th><th>Scale<\/th><th>Normalized<\/th><th>Contribution<\/th>/)
  assert.match(html, /data-value="0.5123456789" title="0.5123456789"/)
  assert.match(html, /data-loss-term="esm2"/)
  assert.match(metricDescription('rosetta_interface_sc', projected), /Higher is preferred/)
  assert.match(metricDescription('rosetta_interface_dg', projected), /Lower is preferred.*REU/)
  loss.base_loss = 2.25
  assert.match(
    renderProteinDesignDashboard({ run: projected }),
    /Original \/ base loss<\/strong> <span data-value="2.25"/
  )
})

test('bounded loss shows fixed calibration, grouped penalties and separate non-selection diagnostics', () => {
  const loss = {
    formula_version: 'bounded-fixed-grouped-v1',
    loss_combination: { calibration_id: '<trial-anchors>' },
    structure_loss: 0.1,
    esm2_contribution: 0.1,
    base_loss: 0.2,
    metric_loss: 0.15,
    loss: 0.35,
    original_base_loss: 2.1,
    legacy_loss: -1.9,
    groups: {
      structure: { budget: 0.5, contribution: 0.1 },
      naturalness: { budget: 0.2, contribution: 0.1 },
      interface: { budget: 0.3, contribution: 0.15 }
    },
    components: {
      plddt: {
        raw: 0.2,
        good: 0,
        bad: 1,
        penalty: 0.2,
        group: 'structure',
        weight: 1,
        normalized_weight: 1,
        effective_weight: 0.5,
        contribution: 0.1
      },
      rosetta_interface_dg: {
        raw: -30.123456789,
        good: -60,
        bad: 0,
        penalty: 0.5,
        group: 'interface',
        weight: 0.5,
        normalized_weight: 1,
        effective_weight: 0.3,
        contribution: 0.15
      }
    },
    esm2_component: {
      raw: -2,
      good: -1,
      bad: -3,
      penalty: 0.5,
      group: 'naturalness',
      weight: 0.1,
      normalized_weight: 1,
      effective_weight: 0.2,
      contribution: 0.1
    }
  }
  const candidate = {
    candidateId: 'bounded',
    objective: 0.35,
    metrics: { loss: 0.35 },
    metadata: { loss, pyrosetta: { status: 'success' } }
  }
  const html = renderProteinDesignDashboard({
    run: { objectiveKey: 'loss', minimize: true, cycles: [{ cycle: 1, candidates: [candidate] }] }
  })
  assert.match(html, /Loss breakdown · bounded · minimize/)
  assert.match(html, /Composite loss \[0, 1\]<\/strong> <span data-value="0.35"/)
  assert.match(html, /Bounded structural \+ ESM2 subtotal<\/strong> <span data-value="0.2"/)
  assert.match(html, /Calibration: <code>&lt;trial-anchors&gt;<\/code>/)
  assert.match(html, /explicitly configured fixed anchors, not universal scientific defaults/)
  assert.match(html, /Penalty = clip\(\(raw − good\) \/ \(bad − good\), 0, 1\)/)
  assert.match(html, /data-loss-group="interface"/)
  assert.match(html, /<th>Within-group weight<\/th><th>Effective weight<\/th>/)
  assert.match(html, /data-value="-30.123456789" title="-30.123456789"/)
  assert.match(html, /data-loss-term="esm2".*?<td>maximize<\/td>/)
  assert.match(html, /Linear diagnostics only — not used for selection/)
  assert.match(html, /Legacy linear composite <span data-value="-1.9"/)
  assert.doesNotMatch(html, /Metric contribution = sign/)
  assert.doesNotMatch(html, /reference -, scale -/)
})

test('minimum ipAE never substitutes a transformed loss term or the distinct ipSAE score', () => {
  const definition = PROPERTY_DEFINITIONS.find(item => item.key === 'min_ipa')
  const candidate = { metrics: { ipsae: 0.8, min_ipsae: 0.7 }, metadata: { loss: { loss_components: { i_pae: 0.5 } } } }
  assert.equal(propertyValue(candidate, definition), null)
  candidate.metrics.min_ipae = 4.2
  assert.equal(propertyValue(candidate, definition), 4.2)
})

test('raw minimum ipAE is displayed once through its canonical property alias', () => {
  const candidate = { candidateId: 'raw-pae', metrics: { min_ipae: 4.23456789, ranking: 0.7 } }
  const run = { cycles: [{ cycle: 0, candidates: [candidate] }] }
  const definitions = propertyDefinitions(null, run)
  assert.equal(definitions.filter(item => item.label === 'Min ipAE').length, 1)
  assert.equal(
    definitions.some(item => item.key === 'metric:min_ipae'),
    false
  )
  assert.equal(
    definitions.some(item => item.key === 'metric:ranking'),
    false
  )
  const definition = definitions.find(item => item.key === 'min_ipa')
  assert.equal(propertyValue(candidate, definition), 4.23456789)
  assert.match(metricDescription(definition.key, run), /Å; raw minimum interchain predicted aligned error/)
  assert.match(renderProteinDesignDashboard({ run }), /data-value="4.23456789" title="4.23456789"/)
  delete candidate.metrics.min_ipae
  candidate.metadata = { min_ipae: { status: 'unavailable', reason: 'Missing <confidence> data' } }
  assert.match(renderProteinDesignDashboard({ run }), /Min ipAE unavailable: Missing &lt;confidence&gt; data/)
})

test('live refresh keeps untouched filter endpoints open as real candidate domains expand', () => {
  let range = refreshTableFilterRange(undefined, { min: 0, max: 1 })
  for (const domain of [
    { min: 6.98, max: 6.98 },
    { min: 6.98, max: 7.28 },
    { min: 6.98, max: 17.2 }
  ]) {
    range = refreshTableFilterRange(range, domain)
    assert.deepEqual(range, { min: domain.min, max: domain.max, domainMin: domain.min, domainMax: domain.max })
    assert.equal(range.min > range.domainMin || range.max < range.domainMax, false)
  }
  assert.deepEqual(refreshTableFilterRange(range, { min: 1, max: 2 }), {
    min: 1,
    max: 2,
    domainMin: 1,
    domainMax: 2
  })
})

test('live refresh preserves deliberately narrowed filter bounds while untouched endpoints follow data', () => {
  const base = { min: 0, max: 1, domainMin: 0, domainMax: 1 }
  const domain = { min: -1, max: 2 }
  assert.deepEqual(refreshTableFilterRange({ ...base, min: 0.3 }, domain), {
    min: 0.3,
    max: 2,
    domainMin: -1,
    domainMax: 2
  })
  assert.deepEqual(refreshTableFilterRange({ ...base, max: 0.7 }, domain), {
    min: -1,
    max: 0.7,
    domainMin: -1,
    domainMax: 2
  })
  assert.deepEqual(refreshTableFilterRange({ ...base, min: 0.3, max: 0.7 }, domain), {
    min: 0.3,
    max: 0.7,
    domainMin: -1,
    domainMax: 2
  })
})

test('metric sliders preserve untouched full-precision bounds and reach both exact endpoints', () => {
  const domain = { min: 0.4623327629429888, max: 0.4774059724547533 }
  const range = refreshTableFilterRange(undefined, domain)
  updateTableFilterRange(range, 'min', 1, 1000)
  assert.equal(range.max, domain.max)
  assert.ok(range.min > domain.min && range.min < domain.max)
  assert.deepEqual(
    [domain.min, domain.max].filter(value => value >= range.min && value <= range.max),
    [domain.max]
  )
  assert.equal(tableFilterSliderPosition(range, 'min', 1000), 1)
  assert.equal(tableFilterSliderPosition(range, 'max', 1000), 1000)
  assert.deepEqual(refreshTableFilterRange(range, domain), range)
  updateTableFilterRange(range, 'min', 0, 1000)
  assert.equal(range.min, domain.min)
  updateTableFilterRange(range, 'max', 999, 1000)
  assert.equal(range.min, domain.min)
  assert.ok(range.max < domain.max)
  updateTableFilterRange(range, 'max', 1000, 1000)
  assert.equal(range.max, domain.max)
  assert.equal(range.min > range.domainMin || range.max < range.domainMax, false)
  const run = { cycles: [{ cycle: 0, candidates: [domain.min, domain.max].map(loss => ({ metrics: { loss } })) }] }
  assert.match(
    renderProteinDesignDashboard({ run }),
    /min="0" max="1000" step="1" value="1000" data-candidate-filter="loss"/
  )
})

test('filter ticks preserve signed and tiny metric ranges, count increments, and crossed bounds', () => {
  for (const domain of [
    { min: -68.22, max: -12.5 },
    { min: 1e-20, max: 2e-20 },
    { min: 7, max: 10 },
    { min: 0, max: 0 }
  ]) {
    const range = refreshTableFilterRange(undefined, domain)
    updateTableFilterRange(range, 'max', 0, 1000)
    assert.equal(range.max, domain.min)
    updateTableFilterRange(range, 'min', 1000, 1000)
    assert.equal(range.min, domain.max)
    assert.equal(range.max, domain.max)
    updateTableFilterRange(range, 'max', 0, 1000)
    assert.equal(range.min, domain.min)
    const unchanged = { ...range }
    updateTableFilterRange(range, 'min', NaN, 1000)
    assert.deepEqual(range, unchanged)
  }
  const count = refreshTableFilterRange(undefined, { min: 7, max: 10 })
  updateTableFilterRange(count, 'min', 1, 3)
  assert.equal(count.min, 8)
  assert.equal(count.max, 10)
  assert.equal(tableFilterSliderPosition(count, 'min', 3), 1)
})

test('cycle leader follows the minimized composite objective even when ranking score disagrees', () => {
  const better = { candidateId: 'lower-composite', objective: -3.5, metrics: { ranking_score: 0.2 } }
  const worse = { candidateId: 'higher-ranking', objective: 1.2, metrics: { ranking_score: 0.9 } }
  const run = { objectiveKey: 'loss', minimize: true, cycles: [{ cycle: 1, candidates: [worse, better] }] }
  assert.equal(candidateGroups(run)[0].candidates[0], better)
  assert.match(renderProteinDesignDashboard({ run }), /Best Loss \(minimize\)/)
  run.minimize = false
  assert.equal(candidateGroups(run)[0].candidates[0], worse)
  run.cycles[0].cycleBestCandidateId = 'lower-composite'
  assert.equal(candidateGroups(run)[0].candidates[0], better)
})

test('scientific property axes include negative, mixed, constant, count and confidence values', () => {
  const samples = [
    ['loss', [-3.2, -0.2, 0.01]],
    ['rosetta_interface_dg', [-68.22165166709135, -12.5]],
    ['rosetta_total_score', [-1044.076175999564, -1000]],
    ['rosetta_interface_dg_per_sasa', [-2.4695456343275235, -2.4695456343275235]],
    ['rosetta_interface_sc', [0.54, 0.63]],
    ['rosetta_interface_sasa', [2301, 2762.518364462969]],
    ['rosetta_interface_hbonds', [0, 24]],
    ['rosetta_interface_unsat_hbonds', [0, 22]],
    ['rosetta_interface_residues', [0, 101]],
    ['plddt', [0.71, 0.88]],
    ['confidence.plddt', [71, 88]],
    ['custom', [0, 0]]
  ]
  for (const [key, values] of samples) {
    const candidates = values.map(value => ({ metrics: { [key]: value } }))
    const definition = { key: `metric:${key}`, paths: [`metrics.${key}`] }
    assert.equal(propertyValue(candidates[0], definition), values[0], key)
    const geometry = parallelGeometry(candidates, [definition])
    const axis = geometry.axes[0]
    assert.ok(axis.max > axis.min, key)
    for (const value of values) {
      assert.ok(value >= axis.min && value <= axis.max, key)
      assert.ok(
        geometry.y(value, axis) >= geometry.top && geometry.y(value, axis) <= geometry.height - geometry.bottom,
        key
      )
    }
    const ticks = propertyTicks(axis)
    assert.equal(ticks[0], axis.max)
    assert.ok(Math.abs(ticks.at(-1) - axis.min) < 1e-9, key)
    if (/hbonds|residues/.test(key)) assert.ok(ticks.every(Number.isInteger), key)
  }
})

function pdbFromPoints(points) {
  return `${points.map(([x, y, z], index) => `ATOM  ${String(index + 1).padStart(5)}  CA  ALA A${String(index + 1).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00 90.00           C`).join('\n')}\nEND\n`
}

test('property-line navigation reveals, activates and scrolls a filtered collapsed candidate', () => {
  const details = { open: false }
  const group = { hidden: true }
  const calls = []
  const table = {
    scrollTop: 200,
    querySelector: () => ({ getBoundingClientRect: () => ({ height: 40 }) }),
    getBoundingClientRect: () => ({ top: 100 }),
    scrollTo: options => calls.push(['scroll', options])
  }
  const row = {
    dataset: { candidateKey: '3:variant' },
    hidden: true,
    closest: selector =>
      ({ '.protein-cycle-group': group, '.protein-cycle-more': details, '.protein-candidate-table': table })[selector],
    click: () => calls.push(['activate', details.open]),
    focus: options => calls.push(['focus', options]),
    getBoundingClientRect: () => ({ top: 500 })
  }
  const view = { value: 'post-filter', dispatchEvent: event => calls.push(['view', view.value, event.type]) }
  const root = {
    querySelectorAll: () => [row],
    querySelector: selector => (selector === '[data-result-view]' ? view : { click: () => calls.push(['reset']) })
  }
  assert.equal(revealCandidateRow(root, 'missing'), false)
  assert.equal(revealCandidateRow(root, '3:variant'), true)
  assert.deepEqual(calls, [
    ['view', 'design', 'change'],
    ['reset'],
    ['activate', true],
    ['focus', { preventScroll: true }],
    ['scroll', { top: 552, left: 0, behavior: 'smooth' }]
  ])
  calls.length = 0
  row.hidden = group.hidden = false
  revealCandidateRow(root, '3:variant')
  assert.equal(
    calls.some(([action]) => action === 'reset'),
    false
  )
})

const candidate = (candidateId, cycle, rankingScore, overrides = {}) => ({
  candidateId,
  cycle,
  sequence: overrides.sequence || `EVQLVESGGGLVQPGGSLRLSCAAS${cycle}${candidateId}`,
  objective: overrides.objective ?? 0.4,
  metrics: {
    loss: overrides.lossCd ?? 0.31,
    iptm: overrides.iptm ?? 0.72,
    ranking_score: rankingScore,
    ipsae: overrides.ipsae ?? 0.58,
    min_ipae: overrides.minIpa ?? 0.23,
    cdr_total_contacts: overrides.contacts ?? 12,
    framework_total_contacts: overrides.frameContacts ?? 2
  },
  metadata: {
    gate_evidence: {
      cdr_total_contacts: overrides.contacts ?? 12,
      framework_total_contacts: overrides.frameContacts ?? 2
    },
    loss: {
      loss_components: { i_pae: overrides.minIpa ?? 0.23 },
      paratope_loss_breakdown: { cdr_contact_loss: overrides.lossCd ?? 0.31 }
    }
  },
  structureArtifactPath: overrides.structureArtifactPath || null
})

const run = {
  taskId: 'task-1',
  target: 'CRLF2',
  computeUrl: 'http://zkln-10:8080',
  computeWorkerId: 'zkln-10',
  status: 'running',
  phase: 'fold',
  cycle: 2,
  totalCycles: 12,
  objectiveKey: 'loss',
  minimize: true,
  candidateCount: 5,
  startTime: '2026-08-23T00:07:22.000Z',
  cycles: [
    {
      cycle: 1,
      candidates: [
        candidate('cycle-1-low', 1, 0.42),
        candidate('cycle-1-best', 1, 0.81, { structureArtifactPath: '/artifacts/cycle-1-best.json' })
      ]
    },
    {
      cycle: 2,
      candidates: [
        candidate('cycle-2-best', 2, 0.92, {
          iptm: 0.84,
          minIpa: 0.12,
          lossCd: 0.18,
          contacts: 17,
          frameContacts: 1,
          structureArtifactPath: '/artifacts/cycle-2-best.json'
        }),
        candidate('cycle-2-mid', 2, 0.63),
        candidate('cycle-2-low', 2, 0.31)
      ]
    }
  ],
  structures: [
    {
      candidateId: 'cycle-2-best',
      cycle: 2,
      artifactPath: '/artifacts/cycle-2-best.json',
      targetChainIds: ['A'],
      binderChainIds: ['D']
    }
  ]
}

test('loads the pinned Molstar viewer with a CDN fallback and interface-focused defaults', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'ui', 'protein-design-candidates.js'), 'utf8')
  assert.match(source, /const MOLSTAR_VERSION = '5\.11\.0'/)
  assert.match(source, /build\/viewer`/)
  assert.match(source, /cdn\.jsdelivr\.net/)
  assert.match(source, /unpkg\.com/)
  assert.match(source, /script\.src = `\$\{baseUrl\}\/molstar\.js`/)
  assert.match(source, /stylesheet\.href = `\$\{baseUrl\}\/molstar\.css`/)
  assert.match(source, /layoutShowSequence: true/)
  assert.match(source, /layoutShowControls: true/)
  assert.match(source, /'plddt-confidence': 'pLDDT'/)
  assert.match(source, /type: 'molecular-surface'/)
  assert.match(source, /type: 'cartoon'/)
  assert.match(source, /Promise\.allSettled/)
  assert.doesNotMatch(source, /3Dmol/)
})

test('rigidly aligns multiple PDB structures by their alpha carbons', () => {
  const referencePoints = [
    [0, 0, 0],
    [2, 0, 0],
    [0, 3, 0],
    [0, 0, 4],
    [2, 3, 1]
  ]
  const movingPoints = referencePoints.map(([x, y, z]) => [-y + 12, x - 7, z + 5])
  const aligned = alignStructureArtifacts([
    {
      candidate: { candidateId: 'reference' },
      structure: { text: pdbFromPoints(referencePoints), format: 'pdb', binder_chain_ids: ['A'] }
    },
    {
      candidate: { candidateId: 'moving' },
      structure: { text: pdbFromPoints(movingPoints), format: 'pdb', binder_chain_ids: ['A'] }
    }
  ])
  const alignedPoints = pdbAlphaCarbons(aligned[1].structure.text, ['A'])
  const rmsd = Math.sqrt(
    alignedPoints.reduce(
      (sum, point, index) =>
        sum + point.reduce((pointSum, value, axis) => pointSum + (value - referencePoints[index][axis]) ** 2, 0),
      0
    ) / referencePoints.length
  )

  assert.ok(rmsd < 0.002, `expected aligned RMSD below 0.002, received ${rmsd}`)
})

test('breaks equal objective ties by ranking score and starts with no comparison selection', () => {
  const groups = candidateGroups(run)
  assert.deepEqual(
    groups.map(group => group.cycle),
    [2, 1]
  )
  assert.deepEqual(
    groups[0].candidates.map(item => item.candidateId),
    ['cycle-2-best', 'cycle-2-mid', 'cycle-2-low']
  )
  assert.deepEqual([...defaultSelectedKeys(run)], [])
})

test('resolves the six requested properties from production evidence aliases', () => {
  const item = run.cycles[1].candidates[0]
  assert.deepEqual(
    PROPERTY_DEFINITIONS.map(definition => definition.label),
    ['ipTM', 'Ranking score', 'Min ipAE', 'Loss', 'Contacts', 'Frame contacts']
  )
  assert.deepEqual(
    PROPERTY_DEFINITIONS.map(definition => propertyValue(item, definition)),
    [0.84, 0.92, 0.12, 0.18, 17, 1]
  )
})

test('renders the candidate table, expandable cycle rows, structure pane, and properties', () => {
  const html = renderProteinDesignDashboard({ runs: [run], selectedRunId: run.taskId, run })

  assert.match(html, /class="protein-candidate-workspace"/)
  assert.match(html, /class="protein-candidate-panel"/)
  assert.match(html, /class="protein-inspector-panel"/)
  assert.match(
    html,
    />Cycle<\/span><span>ID<\/span><span>Target<\/span><span>Sequence<\/span><span>Ranking score<\/span><span>CDR contacts<\/span><span>Min ipAE<\/span>/
  )
  assert.match(html, />Sequence</)
  assert.match(html, />Ranking score</)
  assert.match(html, />Cycle</)
  assert.match(html, /cycle-2-best/)
  assert.match(html, /data-candidate-select="2:cycle-2-best"/)
  assert.doesNotMatch(html, /data-candidate-select="[^"]+" checked/)
  assert.match(html, /data-copy-sequence=/)
  assert.match(html, /View other sequences/)
  assert.match(html, /data-candidate-sort/)
  assert.match(html, /class="protein-candidate-filters"/)
  assert.match(html, /data-candidate-filter="iptm"/)
  assert.match(html, /data-candidate-filter="cdr_contacts"/)
  assert.match(html, /data-table-target="CRLF2"/)
  assert.match(html, /class="protein-overview-metrics"/)
  assert.match(html, /<small>Candidates<\/small><strong>5<\/strong>/)
  assert.match(html, /<small>Best ranking<\/small><strong>0\.92<\/strong>/)
  assert.match(html, /<small>Best ipTM<\/small><strong>0\.84<\/strong>/)
  assert.match(html, /<small>Min ipAE<\/small><strong>0\.12<\/strong>/)
  assert.match(html, /id="proteinStructureViewer"/)
  assert.match(html, /class="protein-plddt-legend"/)
  assert.match(html, /data-structure-color="design" class="is-active"/)
  assert.match(html, /data-structure-color="chain-id"/)
  assert.match(html, /data-structure-color="element-symbol"/)
  assert.match(html, /data-structure-color="plddt-confidence"/)
  assert.match(html, /data-structure-color="sequence-id"/)
  assert.match(html, /data-structure-action="sidechains"/)
  assert.match(html, /data-structure-action="expand"/)
  assert.match(html, /id="proteinPropertiesChart"/)
  assert.match(html, /class="protein-property-picker"/)
  assert.match(html, /\(6\/\d+\)/)
  assert.match(html, /data-property-key="metric:/)
  assert.match(html, />ipTM</)
  assert.match(html, />Min ipAE</)
  assert.match(html, />Loss</)
  assert.match(html, />Frame contacts</)
  assert.doesNotMatch(html, /zkln-10/)
  assert.doesNotMatch(html, /Final selection|Search tree|Live activity|Metric trends/)
})

test('renders configured CDR groups over the sequence and omits them when unavailable', () => {
  const annotated = {
    ...run,
    cdrRegionGroups: {
      D: [
        [2, 3, 4],
        [9, 10],
        [17, 18, 19]
      ]
    },
    cycles: run.cycles.map(cycle => ({
      ...cycle,
      candidates: cycle.candidates.map(item => ({
        ...item,
        metadata: { ...item.metadata, chains: { D: item.sequence } }
      }))
    }))
  }
  const item = annotated.cycles[0].candidates[0]
  const html = renderProteinDesignDashboard({ runs: [annotated], selectedRunId: annotated.taskId, run: annotated })

  assert.deepEqual(
    candidateCdrRegions(item, annotated).map(region => region.label),
    ['CDR1', 'CDR2', 'CDR3']
  )
  assert.match(html, /class="protein-inline-cdr-sequence"/)
  assert.match(html, /class="protein-inline-cdr protein-full-cdr-1"/)
  assert.doesNotMatch(html, /class="protein-annotated-sequence"/)
  assert.match(html, />CDR1</)
  assert.match(html, />CDR2</)
  assert.match(html, />CDR3</)
  assert.match(html, /class="protein-sequence-preview"/)
  assert.match(html, /class="protein-full-sequence-line"/)
  assert.match(html, new RegExp(`data-copy-sequence="${item.sequence}"`))

  const plainHtml = renderProteinDesignDashboard({ runs: [run], selectedRunId: run.taskId, run })
  assert.doesNotMatch(plainHtml, /class="protein-inline-cdr-sequence"/)
  assert.doesNotMatch(plainHtml, /class="protein-annotated-sequence"/)
})

test('keeps long candidate rows compact while exposing the full sequence preview', () => {
  const sequence =
    'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVST'
  const compact = compactSequence(sequence)
  const longSequenceRun = {
    ...run,
    cycles: [{ cycle: 1, candidates: [candidate('long-sequence', 1, 0.8, { sequence })] }]
  }
  const html = renderProteinDesignDashboard({
    runs: [longSequenceRun],
    selectedRunId: run.taskId,
    run: longSequenceRun
  })

  assert.equal(compact, 'MVLSPADKTNVK...ASLDKFLASVST')
  assert.ok(html.includes(compactSequence(sequence, 40, 10)))
  assert.match(html, /<strong>Full sequence<\/strong>/)
  assert.match(html, /135 residues/)
  assert.match(html, /class="protein-full-sequence-residues"/)
  assert.match(html, /repeat\(3,var\(--protein-sequence-cell-width\)\)/)
})

test('uses zero-based rounded axes, equal spacing, and optional last-visible-property coloring', () => {
  const geometry = parallelGeometry(run.cycles.flatMap(cycle => cycle.candidates))
  assert.equal(
    geometry.axes.every(axis => axis.min === 0),
    true
  )
  assert.equal(geometry.axes.find(axis => axis.key === 'iptm').max, 1)
  assert.equal(geometry.axes.find(axis => axis.key === 'contacts').max, 20)
  assert.deepEqual(
    propertyTicks(geometry.axes[0]).map(value => Number(value.toFixed(4))),
    [1, 0.8, 0.6, 0.4, 0.2, 0]
  )
  assert.notEqual(minIpaColor(0.12, 1), minIpaColor(0.23, 1))

  const subset = propertyDefinitions(['iptm', 'loss', 'frame_contacts'])
  const subsetGeometry = parallelGeometry(
    run.cycles.flatMap(cycle => cycle.candidates),
    subset
  )
  assert.deepEqual(
    subsetGeometry.axes.map(axis => axis.key),
    ['iptm', 'loss', 'frame_contacts']
  )
  assert.equal(subsetGeometry.axes[1].x - subsetGeometry.axes[0].x, subsetGeometry.axes[2].x - subsetGeometry.axes[1].x)

  const html = renderParallelCoordinates(run, new Set(), null, PROPERTY_DEFINITIONS, false, 'last-property')
  assert.match(html, /id="proteinPropertyColorGradient"/)
  assert.match(html, /class="protein-property-color-scale"/)
  assert.match(html, /data-color-property="frame_contacts"/)
  assert.doesNotMatch(html, /<text x="-7"/)
  assert.match(html, /Candidates \(2\)/)
  assert.doesNotMatch(html, /Selected candidates/)
  assert.equal((html.match(/class="protein-candidate-line is-all/g) || []).length, 2)
  assert.deepEqual(
    chartCandidates(run, new Set()).map(item => item.candidateId),
    ['cycle-2-best', 'cycle-1-best']
  )
  assert.equal(
    chartCandidates(
      run,
      new Set(run.cycles.flatMap(cycle => cycle.candidates).map(item => `${item.cycle}:${item.candidateId}`))
    ).length,
    5
  )
  assert.doesNotMatch(html, /protein-property-legend-item/)

  const reordered = propertyDefinitions(['frame_contacts', 'ranking_score', 'iptm'])
  const reorderedHtml = renderParallelCoordinates(run, new Set(), null, reordered, false, 'last-property')
  assert.deepEqual(
    reordered.map(definition => definition.key),
    ['frame_contacts', 'ranking_score', 'iptm']
  )
  assert.ok(
    reorderedHtml.indexOf('data-axis-key="frame_contacts"') < reorderedHtml.indexOf('data-axis-key="ranking_score"')
  )
  assert.ok(reorderedHtml.indexOf('data-axis-key="ranking_score"') < reorderedHtml.indexOf('data-axis-key="iptm"'))
  assert.match(reorderedHtml, /data-color-property="iptm"/)
  assert.doesNotMatch(reorderedHtml, /<text x="-7"/)
})

test('defaults to creation-cycle colors independent of metric axes and selected candidates', () => {
  const before = JSON.stringify(run)
  const html = renderParallelCoordinates(run, new Set())
  assert.match(html, /data-color-property="cycle"/)
  assert.match(html, /older, orange/)
  assert.match(html, /newer, purple/)
  assert.match(html, /Age means generation, not elapsed time/)
  const reordered = propertyDefinitions(['loss', 'iptm'])
  const changed = renderParallelCoordinates(run, new Set(['2:cycle-2-mid']), null, reordered)
  const colorFor = (markup, key) => markup.match(new RegExp(`data-line-key="${key}"[^>]*--candidate-color:([^";]+)`))[1]
  assert.equal(colorFor(html, '1:cycle-1-best'), 'rgb(255 122 26)')
  assert.equal(colorFor(html, '2:cycle-2-best'), 'rgb(145 61 224)')
  assert.equal(colorFor(changed, '2:cycle-2-mid'), colorFor(html, '2:cycle-2-best'))
  assert.equal(colorFor(changed, '1:cycle-1-best'), colorFor(html, '1:cycle-1-best'))
  assert.equal(JSON.stringify(run), before)
  const dashboard = renderProteinDesignDashboard({ runs: [run], run })
  assert.match(dashboard, /aria-label="Candidate line color"/)
  assert.match(dashboard, /value="cycle" selected/)
  assert.match(dashboard, /Last visible metric/)
  const metricDashboard = renderProteinDesignDashboard(
    { runs: [run], run },
    null,
    ['iptm', 'contacts'],
    'last-property'
  )
  assert.match(metricDashboard, /value="last-property" selected/)
  assert.match(metricDashboard, /Colored by Contacts/)
})

test('cycle colors handle cycle zero, a single generation, and unknown cycles', () => {
  const one = { ...run, cycles: [{ cycle: 0, candidates: [candidate('zero', 0, 0.8)] }] }
  const html = renderParallelCoordinates(one, new Set())
  assert.match(html, /--candidate-color:rgb\(200 92 125\)/)
  assert.match(html, /Creation cycle: 0 .* to 0 /)
  const unknown = { ...run, cycles: [{ candidates: [candidate('unknown', null, 0.8)] }] }
  assert.match(renderParallelCoordinates(unknown, new Set()), /--candidate-color:#8a8f98/)
  assert.doesNotMatch(renderParallelCoordinates({ ...run, cycles: [] }, new Set()), /NaN|Infinity/)
})

test('adds selected candidates to cycle leaders and animates the full selected path', () => {
  const selected = new Set(['2:cycle-2-mid', '1:cycle-1-best'])
  const html = renderParallelCoordinates(run, selected, '2:cycle-2-mid')

  assert.equal((html.match(/<path class="protein-candidate-line(?:\s|")/g) || []).length, 3)
  assert.match(html, /data-line-key="2:cycle-2-best"/)
  assert.equal((html.match(/class="protein-candidate-line is-muted/g) || []).length, 1)
  assert.equal((html.match(/class="protein-candidate-line is-selected/g) || []).length, 2)
  assert.match(html, /class="protein-candidate-line-control" data-line-key="2:cycle-2-mid"/)
  assert.match(html, /class="protein-candidate-line is-selected is-entering"/)
  assert.match(html, /pathLength="1"/)
  assert.match(html, /Candidates \(3\)/)
  assert.match(html, /Selected candidates \(2\)/)
  assert.match(html, /role="button" tabindex="0" aria-pressed="true"/)
  const enteringPath = html.match(/class="protein-candidate-line is-selected is-entering"[^>]* d="([^"]+)"/)[1]
  assert.equal((enteringPath.match(/ C /g) || []).length, PROPERTY_DEFINITIONS.length - 1)
})

test('shows a stable compact identifier while retaining the full task ID', () => {
  const taskId = 'ed20892707f840f5957c628e4e7d2e79'
  assert.equal(shortTaskId(taskId), 'ed20892707f8')
  const html = renderProteinDesignDashboard({
    runs: [{ ...run, taskId }],
    selectedRunId: taskId,
    run: { ...run, taskId }
  })
  assert.match(html, /<code>ed20892707f8<\/code>/)
})

test('follows the server active-run default unless the user pinned an available run', () => {
  const runs = [{ taskId: 'new-active' }, { taskId: 'old-run' }]

  assert.equal(selectRunId(runs, 'new-active', 'old-run', false), 'new-active')
  assert.equal(selectRunId(runs, 'new-active', 'old-run', true), 'old-run')
  assert.equal(selectRunId([{ taskId: 'new-active' }], 'new-active', 'old-run', true), 'new-active')
})

test('preserves selected candidates, active candidate, expanded cycles, and list scroll', () => {
  const body = { scrollTop: 420, scrollLeft: 300 }
  const cycle = { dataset: { cycle: '2' } }
  const details = { open: true, closest: () => cycle }
  const root = {
    _proteinSelectedKeys: new Set(['2:cycle-2-best', '1:cycle-1-best']),
    _proteinVisiblePropertyKeys: new Set(['iptm', 'ranking_score', 'min_ipa']),
    _proteinPropertyColorMode: 'last-property',
    _proteinActiveCandidateKey: '2:cycle-2-best',
    querySelector: selector => (selector === '.protein-candidate-table' ? body : null),
    querySelectorAll: selector =>
      selector === '.protein-cycle-more[open]' || selector === '.protein-cycle-more' ? [details] : []
  }

  const saved = captureDashboardViewState(root)
  body.scrollTop = 0
  body.scrollLeft = 0
  details.open = false
  restoreDashboardViewState(root, saved)

  assert.equal(body.scrollTop, 420)
  assert.equal(body.scrollLeft, 300)
  assert.deepEqual(saved.selectedKeys, ['2:cycle-2-best', '1:cycle-1-best'])
  assert.deepEqual(saved.visiblePropertyKeys, ['iptm', 'ranking_score', 'min_ipa'])
  assert.equal(saved.propertyColorMode, 'last-property')
  assert.equal(saved.activeCandidateKey, '2:cycle-2-best')
  assert.equal(details.open, true)
})

test('handles cycle zero and missing selection without a mount exception', () => {
  const root = { querySelector: () => null, querySelectorAll: () => [], innerHTML: '' }
  require('../ui/protein-design-candidates').mount(root, { runs: [], run: null })
  assert.match(root.innerHTML, /No protein-design runs/)
  const html = renderProteinDesignDashboard({ runs: [run], run: { ...run, cycle: 0, cycles: [], structures: [] } })
  assert.match(html, /Cycle 0/)
  assert.match(html, /No candidate sequences/)
})

test('does not label objective loss as a missing ranking score', () => {
  const ranking = PROPERTY_DEFINITIONS.find(item => item.key === 'ranking_score')
  assert.equal(propertyValue({ objective: 8.5, metrics: {} }, ranking), null)
})

test('missing scientific measurements break property lines instead of plotting fabricated zeroes', () => {
  const candidate = { candidateId: 'missing-ipae', cycle: 0, metrics: { iptm: 0.8, loss: -2 } }
  const run = { cycles: [{ cycle: 0, candidates: [candidate] }] }
  const definitions = propertyDefinitions(['iptm', 'min_ipa', 'loss'])
  const html = renderParallelCoordinates(run, new Set(), null, definitions)
  assert.match(html, />Unavailable<\/text>/)
  const coordinates = html.match(/class="protein-candidate-line is-all has-missing-values"[^>]* d="([^"]+)"/)[1]
  assert.equal((coordinates.match(/M /g) || []).length, 2)
  assert.doesNotMatch(coordinates, / C /)
})

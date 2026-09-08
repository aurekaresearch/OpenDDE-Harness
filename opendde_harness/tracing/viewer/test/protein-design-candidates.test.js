const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const shell = require('../ui/shell');
const {
  PROPERTY_DEFINITIONS,
  metricLabel,
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
  restoreDashboardViewState,
  selectRunId,
  shortTaskId
} = require('../ui/protein-design-candidates');

test('metric labels use canonical scientific capitalization', () => {
  assert.equal(metricLabel('cdr3_gate_passed'), 'CDR3 gate passed');
  for (const [key, label] of Object.entries({iptm: 'ipTM', ptm: 'pTM', plddt: 'pLDDT', pae: 'PAE', min_ipae: 'Min ipAE', ipsae: 'ipSAE', ranking_score: 'Ranking score', 'confidence.plddt': 'pLDDT'})) {
    assert.equal(metricLabel(key), label);
  }
});

function pdbFromPoints(points) {
  return `${points.map(([x, y, z], index) => `ATOM  ${String(index + 1).padStart(5)}  CA  ALA A${String(index + 1).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00 90.00           C`).join('\n')}\nEND\n`;
}

test('property-line navigation reveals, activates and scrolls a filtered collapsed candidate', () => {
  const details = {open: false};
  const group = {hidden: true};
  const calls = [];
  const table = {
    scrollTop: 200,
    querySelector: () => ({getBoundingClientRect: () => ({height: 40})}),
    getBoundingClientRect: () => ({top: 100}),
    scrollTo: options => calls.push(['scroll', options])
  };
  const row = {
    dataset: {candidateKey: '3:variant'}, hidden: true,
    closest: selector => ({'.protein-cycle-group': group, '.protein-cycle-more': details, '.protein-candidate-table': table})[selector],
    click: () => calls.push(['activate', details.open]),
    focus: options => calls.push(['focus', options]),
    getBoundingClientRect: () => ({top: 500})
  };
  const view = {value: 'post-filter', dispatchEvent: event => calls.push(['view', view.value, event.type])};
  const root = {
    querySelectorAll: () => [row],
    querySelector: selector => selector === '[data-result-view]' ? view : {click: () => calls.push(['reset'])}
  };
  assert.equal(revealCandidateRow(root, 'missing'), false);
  assert.equal(revealCandidateRow(root, '3:variant'), true);
  assert.deepEqual(calls, [
    ['view', 'design', 'change'], ['reset'], ['activate', true],
    ['focus', {preventScroll: true}], ['scroll', {top: 552, left: 0, behavior: 'smooth'}]
  ]);
  calls.length = 0;
  row.hidden = group.hidden = false;
  revealCandidateRow(root, '3:variant');
  assert.equal(calls.some(([action]) => action === 'reset'), false);
});

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
});

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
        candidate('cycle-2-best', 2, 0.92, { iptm: 0.84, minIpa: 0.12, lossCd: 0.18, contacts: 17, frameContacts: 1, structureArtifactPath: '/artifacts/cycle-2-best.json' }),
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
};

test('loads the pinned Molstar viewer with a CDN fallback and interface-focused defaults', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'ui', 'protein-design-candidates.js'), 'utf8');
  assert.match(source, /const MOLSTAR_VERSION = '5\.11\.0'/);
  assert.match(source, /build\/viewer`/);
  assert.match(source, /cdn\.jsdelivr\.net/);
  assert.match(source, /unpkg\.com/);
  assert.match(source, /script\.src = `\$\{baseUrl\}\/molstar\.js`/);
  assert.match(source, /stylesheet\.href = `\$\{baseUrl\}\/molstar\.css`/);
  assert.match(source, /layoutShowSequence: true/);
  assert.match(source, /layoutShowControls: true/);
  assert.match(source, /'plddt-confidence': 'pLDDT'/);
  assert.match(source, /type: 'molecular-surface'/);
  assert.match(source, /type: 'cartoon'/);
  assert.match(source, /Promise\.allSettled/);
  assert.doesNotMatch(source, /3Dmol/);
});

test('rigidly aligns multiple PDB structures by their alpha carbons', () => {
  const referencePoints = [[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4], [2, 3, 1]];
  const movingPoints = referencePoints.map(([x, y, z]) => [-y + 12, x - 7, z + 5]);
  const aligned = alignStructureArtifacts([
    { candidate: { candidateId: 'reference' }, structure: { text: pdbFromPoints(referencePoints), format: 'pdb', binder_chain_ids: ['A'] } },
    { candidate: { candidateId: 'moving' }, structure: { text: pdbFromPoints(movingPoints), format: 'pdb', binder_chain_ids: ['A'] } }
  ]);
  const alignedPoints = pdbAlphaCarbons(aligned[1].structure.text, ['A']);
  const rmsd = Math.sqrt(alignedPoints.reduce((sum, point, index) => sum + point.reduce(
    (pointSum, value, axis) => pointSum + (value - referencePoints[index][axis]) ** 2,
    0
  ), 0) / referencePoints.length);

  assert.ok(rmsd < 0.002, `expected aligned RMSD below 0.002, received ${rmsd}`);
});

test('sorts every cycle by descending ranking score and starts with no comparison selection', () => {
  const groups = candidateGroups(run);
  assert.deepEqual(groups.map((group) => group.cycle), [2, 1]);
  assert.deepEqual(groups[0].candidates.map((item) => item.candidateId), [
    'cycle-2-best',
    'cycle-2-mid',
    'cycle-2-low'
  ]);
  assert.deepEqual([...defaultSelectedKeys(run)], []);
});

test('resolves the six requested properties from production evidence aliases', () => {
  const item = run.cycles[1].candidates[0];
  assert.deepEqual(PROPERTY_DEFINITIONS.map((definition) => definition.label), [
    'ipTM',
    'Ranking score',
    'Min ipAE',
    'Loss',
    'Contacts',
    'Frame contacts'
  ]);
  assert.deepEqual(PROPERTY_DEFINITIONS.map((definition) => propertyValue(item, definition)), [
    0.84,
    0.92,
    0.12,
    0.18,
    17,
    1
  ]);
});

test('renders the candidate table, expandable cycle rows, structure pane, and properties', () => {
  const html = renderProteinDesignDashboard({ runs: [run], selectedRunId: run.taskId, run });

  assert.match(html, /class="protein-candidate-workspace"/);
  assert.match(html, /class="protein-candidate-panel"/);
  assert.match(html, /class="protein-inspector-panel"/);
  assert.match(html, />Cycle<\/span><span>ID<\/span><span>Target<\/span><span>Sequence<\/span><span>Ranking score<\/span><span>CDR contacts<\/span><span>Min ipAE<\/span>/);
  assert.match(html, />Sequence</);
  assert.match(html, />Ranking score</);
  assert.match(html, />Cycle</);
  assert.match(html, /cycle-2-best/);
  assert.match(html, /data-candidate-select="2:cycle-2-best"/);
  assert.doesNotMatch(html, /data-candidate-select="[^"]+" checked/);
  assert.match(html, /data-copy-sequence=/);
  assert.match(html, /View other sequences/);
  assert.match(html, /data-candidate-sort/);
  assert.match(html, /class="protein-candidate-filters"/);
  assert.match(html, /data-candidate-filter="iptm"/);
  assert.match(html, /data-candidate-filter="cdr_contacts"/);
  assert.match(html, /data-table-target="CRLF2"/);
  assert.match(html, /class="protein-overview-metrics"/);
  assert.match(html, /<small>Candidates<\/small><strong>5<\/strong>/);
  assert.match(html, /<small>Best ranking<\/small><strong>0\.92<\/strong>/);
  assert.match(html, /<small>Best ipTM<\/small><strong>0\.84<\/strong>/);
  assert.match(html, /<small>Min ipAE<\/small><strong>0\.12<\/strong>/);
  assert.match(html, /id="proteinStructureViewer"/);
  assert.match(html, /class="protein-plddt-legend"/);
  assert.match(html, /data-structure-color="design" class="is-active"/);
  assert.match(html, /data-structure-color="chain-id"/);
  assert.match(html, /data-structure-color="element-symbol"/);
  assert.match(html, /data-structure-color="plddt-confidence"/);
  assert.match(html, /data-structure-color="sequence-id"/);
  assert.match(html, /data-structure-action="sidechains"/);
  assert.match(html, /data-structure-action="expand"/);
  assert.match(html, /id="proteinPropertiesChart"/);
  assert.match(html, /class="protein-property-picker"/);
  assert.match(html, /\(6\/\d+\)/);
  assert.match(html, /data-property-key="metric:/);
  assert.match(html, />ipTM</);
  assert.match(html, />Min ipAE</);
  assert.match(html, />Loss</);
  assert.match(html, />Frame contacts</);
  assert.doesNotMatch(html, /zkln-10/);
  assert.doesNotMatch(html, /Final selection|Search tree|Live activity|Metric trends/);
});

test('renders configured CDR groups over the sequence and omits them when unavailable', () => {
  const annotated = {
    ...run,
    cdrRegionGroups: { D: [[2, 3, 4], [9, 10], [17, 18, 19]] },
    cycles: run.cycles.map((cycle) => ({
      ...cycle,
      candidates: cycle.candidates.map((item) => ({
        ...item,
        metadata: { ...item.metadata, chains: { D: item.sequence } }
      }))
    }))
  };
  const item = annotated.cycles[0].candidates[0];
  const html = renderProteinDesignDashboard({ runs: [annotated], selectedRunId: annotated.taskId, run: annotated });

  assert.deepEqual(candidateCdrRegions(item, annotated).map((region) => region.label), ['CDR1', 'CDR2', 'CDR3']);
  assert.match(html, /class="protein-inline-cdr-sequence"/);
  assert.match(html, /class="protein-inline-cdr protein-full-cdr-1"/);
  assert.doesNotMatch(html, /class="protein-annotated-sequence"/);
  assert.match(html, />CDR1</);
  assert.match(html, />CDR2</);
  assert.match(html, />CDR3</);
  assert.match(html, /class="protein-sequence-preview"/);
  assert.match(html, /class="protein-full-sequence-line"/);
  assert.match(html, new RegExp(`data-copy-sequence="${item.sequence}"`));

  const plainHtml = renderProteinDesignDashboard({ runs: [run], selectedRunId: run.taskId, run });
  assert.doesNotMatch(plainHtml, /class="protein-inline-cdr-sequence"/);
  assert.doesNotMatch(plainHtml, /class="protein-annotated-sequence"/);
});

test('keeps long candidate rows compact while exposing the full sequence preview', () => {
  const sequence = 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVST';
  const compact = compactSequence(sequence);
  const longSequenceRun = {
    ...run,
    cycles: [{ cycle: 1, candidates: [candidate('long-sequence', 1, 0.8, { sequence })] }]
  };
  const html = renderProteinDesignDashboard({ runs: [longSequenceRun], selectedRunId: run.taskId, run: longSequenceRun });

  assert.equal(compact, 'MVLSPADKTNVK...ASLDKFLASVST');
  assert.ok(html.includes(compactSequence(sequence, 40, 10)));
  assert.match(html, /<strong>Full sequence<\/strong>/);
  assert.match(html, /135 residues/);
  assert.match(html, /class="protein-full-sequence-residues"/);
  assert.match(html, /repeat\(3,var\(--protein-sequence-cell-width\)\)/);
});

test('uses zero-based rounded axes, equal spacing, and colors by the last visible property', () => {
  const geometry = parallelGeometry(run.cycles.flatMap((cycle) => cycle.candidates));
  assert.equal(geometry.axes.every((axis) => axis.min === 0), true);
  assert.equal(geometry.axes.find((axis) => axis.key === 'iptm').max, 1);
  assert.equal(geometry.axes.find((axis) => axis.key === 'contacts').max, 20);
  assert.deepEqual(propertyTicks(geometry.axes[0]).map((value) => Number(value.toFixed(4))), [1, 0.8, 0.6, 0.4, 0.2, 0]);
  assert.notEqual(minIpaColor(0.12, 1), minIpaColor(0.23, 1));

  const subset = propertyDefinitions(['iptm', 'loss', 'frame_contacts']);
  const subsetGeometry = parallelGeometry(run.cycles.flatMap((cycle) => cycle.candidates), subset);
  assert.deepEqual(subsetGeometry.axes.map((axis) => axis.key), ['iptm', 'loss', 'frame_contacts']);
  assert.equal(subsetGeometry.axes[1].x - subsetGeometry.axes[0].x, subsetGeometry.axes[2].x - subsetGeometry.axes[1].x);

  const html = renderParallelCoordinates(run, new Set());
  assert.match(html, /id="proteinPropertyColorGradient"/);
  assert.match(html, /class="protein-property-color-scale"/);
  assert.match(html, /data-color-property="frame_contacts"/);
  assert.doesNotMatch(html, /<text x="-7"/);
  assert.match(html, /Candidates \(2\)/);
  assert.doesNotMatch(html, /Selected candidates/);
  assert.equal((html.match(/class="protein-candidate-line is-all/g) || []).length, 2);
  assert.deepEqual(chartCandidates(run, new Set()).map((item) => item.candidateId), ['cycle-2-best', 'cycle-1-best']);
  assert.equal(chartCandidates(run, new Set(run.cycles.flatMap((cycle) => cycle.candidates).map((item) => `${item.cycle}:${item.candidateId}`))).length, 5);
  assert.doesNotMatch(html, /protein-property-legend-item/);

  const reordered = propertyDefinitions(['frame_contacts', 'ranking_score', 'iptm']);
  const reorderedHtml = renderParallelCoordinates(run, new Set(), null, reordered);
  assert.deepEqual(reordered.map((definition) => definition.key), ['frame_contacts', 'ranking_score', 'iptm']);
  assert.ok(reorderedHtml.indexOf('data-axis-key="frame_contacts"') < reorderedHtml.indexOf('data-axis-key="ranking_score"'));
  assert.ok(reorderedHtml.indexOf('data-axis-key="ranking_score"') < reorderedHtml.indexOf('data-axis-key="iptm"'));
  assert.match(reorderedHtml, /data-color-property="iptm"/);
  assert.doesNotMatch(reorderedHtml, /<text x="-7"/);
});

test('adds selected candidates to cycle leaders and animates the full selected path', () => {
  const selected = new Set(['2:cycle-2-mid', '1:cycle-1-best']);
  const html = renderParallelCoordinates(run, selected, '2:cycle-2-mid');

  assert.equal((html.match(/<path class="protein-candidate-line(?:\s|")/g) || []).length, 3);
  assert.match(html, /data-line-key="2:cycle-2-best"/);
  assert.equal((html.match(/class="protein-candidate-line is-muted/g) || []).length, 1);
  assert.equal((html.match(/class="protein-candidate-line is-selected/g) || []).length, 2);
  assert.match(html, /class="protein-candidate-line-control" data-line-key="2:cycle-2-mid"/);
  assert.match(html, /class="protein-candidate-line is-selected is-entering"/);
  assert.match(html, /pathLength="1"/);
  assert.match(html, /Candidates \(3\)/);
  assert.match(html, /Selected candidates \(2\)/);
  assert.match(html, /role="button" tabindex="0" aria-pressed="true"/);
  const enteringPath = html.match(/class="protein-candidate-line is-selected is-entering"[^>]* d="([^"]+)"/)[1];
  assert.equal((enteringPath.match(/ C /g) || []).length, PROPERTY_DEFINITIONS.length - 1);
});

test('shows a stable compact identifier while retaining the full task ID', () => {
  const taskId = 'ed20892707f840f5957c628e4e7d2e79';
  assert.equal(shortTaskId(taskId), 'ed20892707f8');
  const html = renderProteinDesignDashboard({
    runs: [{ ...run, taskId }],
    selectedRunId: taskId,
    run: { ...run, taskId }
  });
  assert.match(html, /<code>ed20892707f8<\/code>/);
});

test('follows the server active-run default unless the user pinned an available run', () => {
  const runs = [{ taskId: 'new-active' }, { taskId: 'old-run' }];

  assert.equal(selectRunId(runs, 'new-active', 'old-run', false), 'new-active');
  assert.equal(selectRunId(runs, 'new-active', 'old-run', true), 'old-run');
  assert.equal(selectRunId([{ taskId: 'new-active' }], 'new-active', 'old-run', true), 'new-active');
});

test('preserves selected candidates, active candidate, expanded cycles, and list scroll', () => {
  const body = { scrollTop: 420, scrollLeft: 300 };
  const cycle = { dataset: { cycle: '2' } };
  const details = { open: true, closest: () => cycle };
  const root = {
    _proteinSelectedKeys: new Set(['2:cycle-2-best', '1:cycle-1-best']),
    _proteinVisiblePropertyKeys: new Set(['iptm', 'ranking_score', 'min_ipa']),
    _proteinActiveCandidateKey: '2:cycle-2-best',
    querySelector: (selector) => selector === '.protein-candidate-table' ? body : null,
    querySelectorAll: (selector) => selector === '.protein-cycle-more[open]' || selector === '.protein-cycle-more' ? [details] : []
  };

  const saved = captureDashboardViewState(root);
  body.scrollTop = 0;
  body.scrollLeft = 0;
  details.open = false;
  restoreDashboardViewState(root, saved);

  assert.equal(body.scrollTop, 420);
  assert.equal(body.scrollLeft, 300);
  assert.deepEqual(saved.selectedKeys, ['2:cycle-2-best', '1:cycle-1-best']);
  assert.deepEqual(saved.visiblePropertyKeys, ['iptm', 'ranking_score', 'min_ipa']);
  assert.equal(saved.activeCandidateKey, '2:cycle-2-best');
  assert.equal(details.open, true);
});

test('handles cycle zero and missing selection without a mount exception', () => {
  const root = { querySelector: () => null, querySelectorAll: () => [], innerHTML: '' };
  require('../ui/protein-design-candidates').mount(root, { runs: [], run: null });
  assert.match(root.innerHTML, /No protein-design runs/);
  const html = renderProteinDesignDashboard({runs:[run], run:{...run, cycle:0, cycles:[], structures:[]}});
  assert.match(html, /Cycle 0/);
  assert.match(html, /No candidate sequences/);
});

test('does not label objective loss as a missing ranking score', () => {
  const ranking = PROPERTY_DEFINITIONS.find(item => item.key === 'ranking_score');
  assert.equal(propertyValue({objective: 8.5, metrics:{}}, ranking), null);
});

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const shell = require('../ui/shell');

test('theme control restores the saved mode and persists both toggle directions', () => {
  const vm = require('node:vm');
  const source = fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8');
  const themeSource = source.slice(source.indexOf('function applyTheme(theme)'), source.lastIndexOf('applyStaticI18n();'));
  const attributes = {};
  let click;
  const button = {setAttribute: (key, value) => { attributes[key] = value; }, addEventListener: (_, handler) => { click = handler; }};
  const document = {documentElement: {dataset: {}}, getElementById: id => id === 'themeButton' ? button : null, querySelectorAll: () => []};
  const saved = new Map([['tracing:theme', 'dark']]);
  vm.runInNewContext(themeSource, {document, elements: {}, localStorage: {getItem: key => saved.get(key), setItem: (key, value) => saved.set(key, value)}});
  assert.equal(document.documentElement.dataset.theme, 'dark');
  assert.equal(attributes['aria-pressed'], 'true');
  click();
  assert.equal(document.documentElement.dataset.theme, 'light');
  assert.equal(saved.get('tracing:theme'), 'light');
  click();
  assert.equal(saved.get('tracing:theme'), 'dark');
  assert.match(shell, /id="themeButton"/);
});

test('tracing board is English-only without language controls or saved language selection', () => {
  const app = fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8');
  assert.doesNotMatch(shell, /lang-switch|lang-pill|data-lang=/);
  assert.doesNotMatch(app, /tracing:lang|state\.lang|setLang|zh-CN|\bzh:\s*\{/);
  assert.match(app, /document\.documentElement\.lang = 'en'/);
  assert.match(app, /const value = I18N\.en\[key\]/);
});
const {
  buildTreeLayout,
  captureDashboardViewState,
  orderMetricNames,
  renderActivity,
  renderProteinDesignDashboard,
  renderMarkdown,
  restoreDashboardViewState,
  selectRunId,
  shortTaskId
} = require('../ui/protein-design');

const run = {
  taskId: 'task-1',
  target: 'CRLF2',
  computeUrl: 'http://zkln-10:8080',
  computeWorkerId: 'zkln-10',
  status: 'running',
  phase: 'fold',
  cycle: 12,
  totalCycles: 100,
  objectiveKey: 'loss',
  minimize: true,
  bestCandidateId: 'candidate-2',
  bestObjective: 0.31,
  startTime: '2026-08-23T00:07:22.000Z',
  endTime: '2026-08-23T00:18:44.000Z',
  candidateCount: 2,
  failureCount: 1,
  metricNames: ['z_metric', 'iptm', 'loss', 'confidence.plddt'],
  series: {
    loss: {
      cycleBest: [{ cycle: 0, candidateId: 'candidate-1', value: 0.42 }],
      globalBest: [{ cycle: 0, candidateId: 'candidate-2', value: 0.31 }]
    },
    iptm: {
      cycleBest: [{ cycle: 0, candidateId: 'candidate-1', value: 0.72 }],
      globalBest: [{ cycle: 0, candidateId: 'candidate-2', value: 0.68 }]
    },
    'confidence.plddt': { cycleBest: [], globalBest: [] },
    z_metric: { cycleBest: [], globalBest: [] }
  },
  structures: [
    {
      candidateId: 'candidate-2',
      sequence: 'EVQLVESGGGLVQPGGSLRLSCAAS',
      cycle: 0,
      artifactPath: '/artifacts/best.json',
      objective: 0.31,
      targetChainIds: ['A'],
      binderChainIds: ['D'],
      inPopulation: true,
      admitted: true,
      populationAction: 'inserted',
      gatePassed: true,
      gateMetrics: {
        cdr_contact_fraction: 0.82,
        cdr_total_contacts: 12,
        framework_total_contacts: 2
      },
      gateEvidence: {
        reason: null,
        framework_contact_residue_ids: [{ chain: 'D', residue_id: 19 }]
      },
      skillId: 'cdr-point-mutation'
    }
  ],
  populationCandidateIds: ['candidate-2'],
  postFilter: {
    enabled: true,
    executed: true,
    mode: 'agent',
    status: 'completed',
    strategySummary: 'Prefer intended CDR contacts.',
    topK: 1,
    refoldEnabled: true,
    refoldedCandidateCount: 2,
    eligibleCandidateCount: 1,
    selectedCandidateIds: ['candidate-2'],
    decisions: [{
      candidateId: 'candidate-2', rank: 1, score: 0.94, objective: 0.31,
      metrics: { iptm: 0.81, plddt: 0.88, cdr_contact_fraction: 0.84 },
      passFilter: true, hardEligible: true, rationale: 'Best CDR geometry.',
      strengths: ['CDR contact'], risks: [],
      structureArtifactPath: '/artifacts/refolded-best.json',
      targetChainIds: ['A'], binderChainIds: ['D']
    }]
  },
  tree: {
    nodes: [
      { id: 'cycle:0', kind: 'cycle', cycle: 0, label: 'Cycle 0' },
      {
        id: 'candidate:candidate-2',
        kind: 'candidate',
        parentId: 'cycle:0',
        candidateId: 'candidate-2',
        cycle: 0,
        objective: 0.31,
        skillId: 'cdr-point-mutation'
      }
    ],
    edges: [{ source: 'cycle:0', target: 'candidate:candidate-2' }]
  }
};

test('shell exposes API Calls, Traces, and Protein design workspaces', () => {
  assert.match(shell, /data-app-view="api"/);
  assert.match(shell, /data-app-view="trace"/);
  assert.match(shell, /data-app-view="protein-design"/);
  assert.match(shell, /id="proteinDesignScene"/);
  assert.match(shell, /<script src="\/protein-design\.js"><\/script>/);
});

test('loads the molecular viewer from the bundled dashboard asset', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'ui', 'protein-design.js'), 'utf8');
  assert.match(source, /script\.src = '\/vendor\/3Dmol-min\.js'/);
  assert.doesNotMatch(source, /https:\/\/3dmol\.org/);
});

test('orders the configured objective before common metrics and remaining metrics', () => {
  assert.deepEqual(
    orderMetricNames(run.metricNames, run.objectiveKey),
    ['loss', 'iptm', 'confidence.plddt', 'z_metric']
  );
});

test('renders run controls metric points structure controls and candidate tree', () => {
  const html = renderProteinDesignDashboard({
    runs: [run],
    selectedRunId: 'task-1',
    run
  });

  assert.match(html, /<details class="protein-run-picker" id="proteinDesignRunPicker">/);
  assert.match(html, /data-run-id="task-1"/);
  assert.match(html, /title="task-1">task-1<\/code>/);
  assert.match(html, /running · 2026-/);
  assert.match(html, /CRLF2/);
  assert.doesNotMatch(html, /class="protein-stats"/);
  assert.doesNotMatch(html, /<text y="28"/);
  assert.match(html, /protein-tree-detail/);
  assert.match(html, /data-metric="loss"/);
  assert.match(html, /class="protein-metric-explorer"/);
  assert.doesNotMatch(html, /data-metric-select/);
  assert.doesNotMatch(html, /data-metric-focus-panel/);
  assert.match(html, /Current cycle best/);
  assert.match(html, /Global best/);
  assert.match(html, /data-cycle="0"/);
  assert.match(html, /candidate-2 · 0\.31/);
  assert.match(html, /id="proteinStructureViewer"/);
  assert.match(html, /data-candidate-search/);
  assert.match(html, /protein-properties-pane/);
  assert.match(html, /data-structure-color="plddt-confidence"/);
  assert.match(html, /data-structure-action="expand"/);
  assert.match(html, /data-target-chain-ids="\[&quot;A&quot;\]"/);
  assert.match(html, /data-binder-chain-ids="\[&quot;D&quot;\]"/);
  assert.match(html, /data-cycle="0"/);
  assert.match(html, /class="protein-tree-svg"/);
  assert.match(html, /class="protein-tree-legend"/);
  assert.match(html, /data-node-id="candidate:candidate-2"/);
  assert.match(html, /<option value="post-filter">Post-filter<\/option>/);
  assert.ok(html.indexOf('Prefer intended CDR contacts') < html.indexOf('protein-design-details'));
  assert.doesNotMatch(html, /Selected candidate evidence|Finalizing design artifacts and traces/);
  assert.match(html, /Prefer intended CDR contacts/);
  assert.match(html, /Best CDR geometry/);
  assert.doesNotMatch(html, /Filter score/);
  assert.match(html, /Successfully refolded/);
  assert.match(html, /data-post-filter-structure/);
  assert.match(html, /class="protein-filter-overview"/);
  assert.match(html, /class="protein-filter-structure-panel"/);
  assert.match(html, /data-structure-path="\/artifacts\/refolded-best\.json"/);
  assert.match(html, /data-candidate-id="candidate-2"/);
  assert.match(html, /ipTM/);
  assert.match(html, /pLDDT/);
  assert.match(html, /0\.81/);
});

test('keeps the post-filter panel empty when the stage was not enabled', () => {
  const html = renderProteinDesignDashboard({
    runs: [run], selectedRunId: 'task-1', run: { ...run, postFilter: null }
  });
  assert.match(html, /No post-filter stage data for this run/);
  assert.doesNotMatch(html, /Best CDR geometry/);
});

test('shows a stable compact identifier while retaining the full task ID', () => {
  const taskId = 'ed20892707f840f5957c628e4e7d2e79';
  assert.equal(shortTaskId(taskId), 'ed20892707f8');
  const html = renderProteinDesignDashboard({
    runs: [{ ...run, taskId }],
    selectedRunId: taskId,
    run: { ...run, taskId }
  });
  assert.match(html, /title="ed20892707f840f5957c628e4e7d2e79">ed20892707f8<\/code>/);
});

test('tree layout aligns nodes from the same cycle and remains bounded', () => {
  const layout = buildTreeLayout(run.tree, { width: 800, rowHeight: 70 });
  const root = layout.nodes.find((node) => node.id === 'cycle:0');
  const child = layout.nodes.find((node) => node.id === 'candidate:candidate-2');

  assert.equal(child.y, root.y);
  assert.ok(root.x >= 24 && root.x <= 776);
  assert.ok(child.x >= 24 && child.x <= 776);
  assert.equal(layout.edges[0].source.id, 'cycle:0');
  assert.equal(layout.edges[0].target.id, 'candidate:candidate-2');
});

test('tree layout separates actual cycles rather than parent depth', () => {
  const denseTree = {
    nodes: [
      { id: 'candidate:initial', kind: 'initial', candidateId: 'initial', cycle: -1 },
      ...Array.from({ length: 24 }, (_, index) => ({
        id: `candidate:c${index}`,
        kind: 'candidate',
        parentId: 'candidate:initial',
        candidateId: `c${index}`,
        cycle: index
      }))
    ],
    edges: Array.from({ length: 24 }, (_, index) => ({
      source: 'candidate:initial',
      target: `candidate:c${index}`
    }))
  };
  const layout = buildTreeLayout(denseTree);
  assert.deepEqual(layout.levels, Array.from({length:25}, (_,i) => i-1));
  assert.ok(layout.nodes.find(node => node.id === 'candidate:c23').y > layout.nodes.find(node => node.id === 'candidate:c0').y);
});

test('follows the server active-run default unless the user pinned an available run', () => {
  const runs = [{ taskId: 'new-active' }, { taskId: 'old-run' }];

  assert.equal(selectRunId(runs, 'new-active', 'old-run', false), 'new-active');
  assert.equal(selectRunId(runs, 'new-active', 'old-run', true), 'old-run');
  assert.equal(selectRunId([{ taskId: 'new-active' }], 'new-active', 'old-run', true), 'new-active');
});

test('disables Open trace when the referenced trace is no longer available', () => {
  const html = renderProteinDesignDashboard({
    runs: [run],
    selectedRunId: 'task-1',
    run: { ...run, traceId: 'trace-task-1', traceAvailable: false }
  });

  assert.match(html, /id="proteinOpenTrace"[^>]*disabled/);
  assert.match(html, /Trace is no longer available/);
});

test('renders live activity summaries with expandable full agent IO', () => {
  const event = {
    event_id: 'event-1',
    task_id: 'task-1',
    event_type: 'agent',
    status: 'completed',
    cycle: 4,
    phase: 'design',
    actor: 'design',
    skill: 'cdr-point-mutation',
    duration_ms: 1200,
    candidate_count: 8,
    summary: 'design agent completed',
    input_payload: { prompt: 'full prompt' },
    output_payload: { skill_id: 'cdr-point-mutation' }
  };
  const dashboard = renderProteinDesignDashboard({
    runs: [run], selectedRunId: 'task-1', run: { ...run, events: [event] }
  });
  assert.doesNotMatch(dashboard, /Live activity/);
  assert.match(dashboard, /Select a step above/);
  const html = renderActivity([event], { deferPayloads: true });
  assert.match(html, /Cycle 4/);
  assert.match(html, /cdr-point-mutation/);
  assert.match(html, /<details/);
  assert.match(html, />Input</);
  assert.match(html, />Output</);
  assert.match(html, /Expand to load full Input \/ Output/);
  assert.doesNotMatch(html, /full prompt/);
  const expanded = renderActivity([event]);
  assert.match(expanded, /class="protein-activity-io-grid"/);
  assert.match(expanded, /full prompt/);
  assert.match(expanded, /Skill id/);
});

test('shows the design loop below results without the redundant thinking banner', () => {
  const events = [
    {
      event_id: 'event-design', status: 'completed', cycle: 7, phase: 'design',
      event_type: 'agent', actor: 'Design Agent', skill: 'cdr-point-mutation',
      timestamp: '2026-08-23T00:10:00.000Z', duration_ms: 900
    },
    {
      event_id: 'event-fold', status: 'started', cycle: 7, phase: 'fold',
      event_type: 'compute', actor: 'OpenDDE', candidate_count: 8,
      timestamp: '2026-08-23T00:10:01.000Z'
    }
  ];
  const html = renderProteinDesignDashboard({
    runs: [run], selectedRunId: 'task-1', run: { ...run, cycle: 8, events }
  });

  assert.match(html, /Cycle 8 of 100/);
  assert.match(html, /aria-label="Design, fold, reflection, and parent selection loop"/);
  assert.match(html, /data-loop-stage="fold"[^>]*aria-current="step"/);
  assert.match(html, /protein-loop-stage-fold is-current/);
  assert.match(html, /protein-loop-stage-design is-completed/);
  assert.match(html, /data-loop-stage="reflection"/);
  assert.match(html, /data-loop-stage="parent_selection"/);
  assert.match(html, /Current search/);
  assert.doesNotMatch(html, /Live activity/);
  assert.match(html, /protein-loop-connectors/);
  assert.match(html, /data-loop-cycle/);
  assert.match(html, /protein-process-grid/);
  assert.ok(html.indexOf('<h2>Search tree') < html.indexOf('<h2>Metric trends'));
});

test('uses the OpenDDE light purple visual system for the tracing board', () => {
  const css = fs.readFileSync(path.resolve(__dirname, '..', 'ui', 'app.css'), 'utf8');
  assert.match(css, /color-scheme:\s*light/);
  assert.match(css, /--bg:\s*#f7f9fc/);
  assert.match(css, /--accent:\s*#6c43c9/);
  assert.match(css, /\.protein-design-loop\s*\{/);
});

test('renders common Markdown safely in agent payloads', () => {
  const html = renderMarkdown('# Result\n\n- **Improved** `iptm`\n\n```json\n{"ok": true}\n```\n<script>alert(1)</script>');
  assert.match(html, /<h3>Result<\/h3>/);
  assert.match(html, /<strong>Improved<\/strong>/);
  assert.match(html, /<code>iptm<\/code>/);
  assert.match(html, /protein-markdown-code/);
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;/);
});

test('preserves dashboard and structure-list scroll state across refresh renders', () => {
  const classes = new Set(['is-expanded']);
  const structureList = { scrollTop: 731, scrollLeft: 4 };
  const filter = { value: 'cycle 8' };
  const style = { value: 'stick' };
  const section = { classList: { contains: (name) => classes.has(name), add: (name) => classes.add(name) } };
  const spinClasses = new Set(['is-active']);
  const spin = { classList: { contains: (name) => spinClasses.has(name), toggle: (name, active) => active ? spinClasses.add(name) : spinClasses.delete(name) } };
  const expand = { textContent: 'Expand' };
  let transform = 'translate(12 8) scale(1.2)';
  const tree = { getAttribute: () => transform, setAttribute: (_name, value) => { transform = value; } };
  const runPicker = { open: true };
  const runPickerMenu = { scrollTop: 248 };
  const activityList = { scrollTop: 391, scrollLeft: 3 };
  const activityDetails = { open: true, dataset: { persistKey: 'activity:event-1' } };
  const payloadScroll = { scrollTop: 177, dataset: { activityScrollKey: 'event-1:output' } };
  const candidates = [{ dataset: { candidateId: 'c8', cycle: '8' }, hidden: false }];
  const selectors = {
    '.protein-structure-list': structureList,
    '#proteinStructureFilter': filter,
    '#proteinStructureStyle': style,
    '.protein-structure-section': section,
    '[data-structure-action="spin"]': spin,
    '[data-structure-action="expand"]': expand,
    '.protein-tree-viewport': tree,
    '#proteinDesignRunPicker': runPicker,
    '.protein-run-picker-menu': runPickerMenu,
    '.protein-activity-list': activityList
  };
  const root = {
    scrollTop: 512,
    scrollLeft: 2,
    querySelector: (selector) => selectors[selector] || null,
    querySelectorAll: (selector) => {
      if (selector === '.protein-structure-option') return candidates;
      if (selector === 'details[data-persist-key][open]') return [activityDetails];
      if (selector === 'details[data-persist-key]') return [activityDetails];
      if (selector === '[data-activity-scroll-key]') return [payloadScroll];
      return [];
    }
  };

  const saved = captureDashboardViewState(root);
  root.scrollTop = 0;
  structureList.scrollTop = 0;
  filter.value = '';
  style.value = 'cartoon';
  classes.clear();
  spinClasses.clear();
  transform = '';
  runPicker.open = false;
  runPickerMenu.scrollTop = 0;
  activityList.scrollTop = 0;
  activityList.scrollLeft = 0;
  activityDetails.open = false;
  payloadScroll.scrollTop = 0;
  restoreDashboardViewState(root, saved);

  assert.equal(root.scrollTop, 512);
  assert.equal(structureList.scrollTop, 731);
  assert.equal(filter.value, 'cycle 8');
  assert.equal(style.value, 'stick');
  assert.equal(classes.has('is-expanded'), true);
  assert.equal(spinClasses.has('is-active'), true);
  assert.equal(expand.textContent, 'Close');
  assert.equal(transform, 'translate(12 8) scale(1.2)');
  assert.equal(runPicker.open, true);
  assert.equal(runPickerMenu.scrollTop, 248);
  assert.equal(activityList.scrollTop, 391);
  assert.equal(activityList.scrollLeft, 3);
  assert.equal(activityDetails.open, true);
  assert.equal(payloadScroll.scrollTop, 177);
  assert.equal(candidates[0].hidden, false);
});


test('post-filter failure renders failed status and no synthetic score', () => {
  const html = renderProteinDesignDashboard({
    runs: [run], selectedRunId: 'task-1',
    run: { ...run, postFilter: {
      ...run.postFilter, mode: 'failed', status: 'failed',
      postFilterError: 'agent unavailable', selectedCandidateIds: [],
      refoldedCandidateCount: 0, eligibleCandidateCount: 0, decisions: []
    } }
  });
  assert.match(html, /<span>Status<\/span><strong>Failed<\/strong>/);
  assert.match(html, /agent unavailable/);
  assert.doesNotMatch(html, /Fallback|Filter score/);
});

(function initProteinDesignDashboard(root, factory) {
  if (root && !document.querySelector('link[href="/protein-design.css"]')) {
    const stylesheet = document.createElement('link');
    stylesheet.rel = 'stylesheet';
    stylesheet.href = '/protein-design.css';
    document.head.appendChild(stylesheet);
  }
  if (root && !root.ProteinDesignCandidates) {
    const script = document.createElement('script');
    script.src = '/protein-design-candidates.js';
    script.onload = () => {
      root.ProteinDesignDashboard = factory();
      root.dispatchEvent(new Event('protein-design-ready'));
    };
    document.head.appendChild(script);
    return;
  }
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ProteinDesignDashboard = api;
})(typeof window === 'undefined' ? null : window, function buildProteinDesignDashboard() {
  const candidatesDashboard = typeof module === 'object' && module.exports
    ? require('./protein-design-candidates')
    : window.ProteinDesignCandidates;
  const COMMON_METRICS = [
    /^iptm$/i,
    /plddt/i,
    /ipsae/i,
    /contact|i_con/i,
    /loss/i
  ];
  let threeDmolPromise = null;

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatNumber(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude >= 10000 || magnitude < 0.001)) return value.toExponential(3);
    return Number(value.toFixed(5)).toString();
  }

  function formatTimestamp(value) {
    const parsed = Date.parse(value || '');
    if (!Number.isFinite(parsed)) return 'Time unavailable';
    const date = new Date(parsed);
    const pad = (part) => String(part).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }

  function shortTaskId(value) {
    const taskId = String(value || 'unknown');
    return taskId.length <= 12 ? taskId : taskId.slice(0, 12);
  }

  function humanizeFieldName(value) {
    return candidatesDashboard.metricLabel(value);
  }

  function renderInlineMarkdown(value) {
    return escapeHtml(value)
      .replace(/`([^`\n]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
      .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
      .replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  }

  function renderMarkdown(value) {
    const lines = String(value ?? '').replace(/\r\n?/g, '\n').split('\n');
    const html = [];
    let codeLanguage = '';
    let codeLines = null;
    for (const line of lines) {
      const fence = line.match(/^\s*```([^`]*)$/);
      if (fence) {
        if (codeLines === null) {
          codeLanguage = fence[1].trim();
          codeLines = [];
        } else {
          html.push(`<pre class="protein-markdown-code"${codeLanguage ? ` data-language="${escapeHtml(codeLanguage)}"` : ''}><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
          codeLanguage = '';
          codeLines = null;
        }
        continue;
      }
      if (codeLines !== null) {
        codeLines.push(line);
        continue;
      }
      if (!line.trim()) {
        html.push('<div class="protein-markdown-spacer" aria-hidden="true"></div>');
        continue;
      }
      const heading = line.match(/^\s*(#{1,4})\s+(.+)$/);
      if (heading) {
        const level = Math.min(6, heading[1].length + 2);
        html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }
      const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
      if (unordered) {
        html.push(`<div class="protein-markdown-list-item"><span aria-hidden="true">•</span><div>${renderInlineMarkdown(unordered[1])}</div></div>`);
        continue;
      }
      const ordered = line.match(/^\s*(\d+)[.)]\s+(.+)$/);
      if (ordered) {
        html.push(`<div class="protein-markdown-list-item"><span>${escapeHtml(ordered[1])}.</span><div>${renderInlineMarkdown(ordered[2])}</div></div>`);
        continue;
      }
      const quote = line.match(/^\s*>\s?(.*)$/);
      if (quote) {
        html.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
        continue;
      }
      html.push(`<p>${renderInlineMarkdown(line)}</p>`);
    }
    if (codeLines !== null) {
      html.push(`<pre class="protein-markdown-code"${codeLanguage ? ` data-language="${escapeHtml(codeLanguage)}"` : ''}><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
    }
    return html.join('');
  }

  function renderPayloadValue(value) {
    if (typeof value === 'string') {
      return `<div class="protein-markdown">${renderMarkdown(value)}</div>`;
    }
    if (value === null || value === undefined) {
      return '<div class="protein-activity-empty">Not captured</div>';
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return `<code class="protein-payload-scalar">${escapeHtml(value)}</code>`;
    }
    return `<pre class="protein-json">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
  }

  function renderPayloadPanel(label, payload, eventId, emptyLabel) {
    let body = `<div class="protein-activity-empty">${escapeHtml(emptyLabel)}</div>`;
    if (payload !== null && payload !== undefined) {
      if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
        const fields = Object.entries(payload).map(([key, value]) => `
          <section class="protein-payload-field">
            <h5>${escapeHtml(humanizeFieldName(key))}</h5>
            ${renderPayloadValue(value)}
          </section>`).join('');
        body = fields || '<div class="protein-activity-empty">Empty object</div>';
      } else {
        body = renderPayloadValue(payload);
      }
    }
    return `<section class="protein-activity-payload">
      <h4>${escapeHtml(label)}</h4>
      <div class="protein-activity-payload-scroll" data-activity-scroll-key="${escapeHtml(`${eventId}:${label.toLowerCase()}`)}">${body}</div>
    </section>`;
  }

  function renderRunPicker(runs, selectedRunId) {
    const selected = runs.find((item) => item.taskId === selectedRunId) || runs[0];
    const selectedTime = selected?.startTime || selected?.endTime || null;
    const options = runs.map((item) => {
      const activeMark = item.active ? '<span class="protein-run-picker-active" aria-label="Active run"></span>' : '';
      const worker = item.computeWorkerId || item.computeUrl || 'default';
      const endLabel = item.active ? 'Updated' : 'Ended';
      const endValue = item.endTime || item.startTime;
      return `<button type="button" class="protein-run-picker-option${item.taskId === selectedRunId ? ' is-selected' : ''}" data-run-id="${escapeHtml(item.taskId)}" role="option" aria-selected="${item.taskId === selectedRunId ? 'true' : 'false'}">
        <span class="protein-run-picker-option-main">${activeMark}<strong>${escapeHtml(item.target || 'unknown')}</strong><code class="protein-run-picker-id" title="${escapeHtml(item.taskId)}">${escapeHtml(shortTaskId(item.taskId))}</code><span class="protein-run-status protein-status-${escapeHtml(item.status)}">${escapeHtml(item.status || 'unknown')}</span></span>
        <span class="protein-run-picker-option-meta"><time datetime="${escapeHtml(item.startTime || '')}">Started ${escapeHtml(formatTimestamp(item.startTime))}</time><span>${escapeHtml(endLabel)} ${escapeHtml(formatTimestamp(endValue))}</span></span>
        <span class="protein-run-picker-option-meta"><span>${escapeHtml(worker)}</span><span>Task ID · ${escapeHtml(shortTaskId(item.taskId))}</span></span>
      </button>`;
    }).join('');
    return `<details class="protein-run-picker" id="proteinDesignRunPicker" data-persist-key="run-picker">
      <summary aria-label="Select protein-design run">
        <span class="protein-run-picker-label">Run</span>
        <span class="protein-run-picker-current"><strong>${escapeHtml(selected?.target || 'unknown')}</strong><code class="protein-run-picker-id" title="${escapeHtml(selected?.taskId || '')}">${escapeHtml(shortTaskId(selected?.taskId))}</code><span>${escapeHtml(selected?.status || 'unknown')}</span></span>
        <time datetime="${escapeHtml(selectedTime || '')}">${escapeHtml(formatTimestamp(selectedTime))}</time>
        <span class="protein-run-picker-chevron" aria-hidden="true">⌄</span>
      </summary>
      <div class="protein-run-picker-menu" role="listbox" aria-label="Protein-design runs">${options}</div>
    </details>`;
  }

  function orderMetricNames(metricNames, objectiveKey) {
    const names = [...new Set(metricNames || [])];
    const result = [];
    if (names.includes(objectiveKey)) result.push(objectiveKey);
    for (const matcher of COMMON_METRICS) {
      names
        .filter((name) => name !== objectiveKey && matcher.test(name) && !result.includes(name))
        .sort((a, b) => a.localeCompare(b))
        .forEach((name) => result.push(name));
    }
    names
      .filter((name) => !result.includes(name))
      .sort((a, b) => a.localeCompare(b))
      .forEach((name) => result.push(name));
    return result;
  }

  function selectRunId(runs, serverSelectedRunId, currentRunId, pinned) {
    const available = new Set((runs || []).map((run) => run.taskId));
    if (pinned && available.has(currentRunId)) return currentRunId;
    if (available.has(serverSelectedRunId)) return serverSelectedRunId;
    return runs?.[0]?.taskId || null;
  }

  function chartGeometry(series, width = 640, height = 190) {
    const points = [...(series?.cycleBest || []), ...(series?.globalBest || [])];
    if (!points.length) return null;
    const cycles = points.map((point) => point.cycle);
    const values = points.map((point) => point.value);
    const minCycle = Math.min(...cycles);
    const maxCycle = Math.max(...cycles);
    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    if (minValue === maxValue) {
      const pad = Math.abs(minValue || 1) * 0.05;
      minValue -= pad;
      maxValue += pad;
    }
    const left = 54;
    const right = 18;
    const top = 18;
    const bottom = 34;
    const x = (cycle) =>
      left + ((cycle - minCycle) / Math.max(1, maxCycle - minCycle)) * (width - left - right);
    const y = (value) =>
      top + ((maxValue - value) / Math.max(Number.EPSILON, maxValue - minValue)) * (height - top - bottom);
    return { width, height, left, right, top, bottom, minCycle, maxCycle, minValue, maxValue, x, y };
  }

  function renderSeries(points, geometry, className) {
    if (!points?.length) return '';
    const line = points.map((point) => `${geometry.x(point.cycle)},${geometry.y(point.value)}`).join(' ');
    const circles = points
      .map((point) => {
        const label = `Cycle ${point.cycle} · ${point.candidateId} · ${formatNumber(point.value)}`;
        return `<circle class="${className}-point" cx="${geometry.x(point.cycle)}" cy="${geometry.y(point.value)}" r="2" data-cycle="${point.cycle}" data-candidate-id="${escapeHtml(point.candidateId)}" data-value="${point.value}" tabindex="0"><title>${escapeHtml(label)}</title></circle>`;
      })
      .join('');
    return `<polyline class="${className}-line" points="${line}" fill="none"/>${circles}`;
  }

  function renderMetricPlot(metric, series, options = {}) {
    const geometry = chartGeometry(series, options.width, options.height);
    if (!geometry) return '';
    return `<div class="protein-chart-wrap${options.focus ? ' protein-chart-focus-wrap' : ''}">
          <svg class="protein-metric-svg" viewBox="0 0 ${geometry.width} ${geometry.height}" role="img" aria-label="${escapeHtml(humanizeFieldName(metric))} by cycle">
            <line class="protein-axis" x1="${geometry.left}" x2="${geometry.left}" y1="${geometry.top}" y2="${geometry.height - geometry.bottom}"/>
            <line class="protein-axis" x1="${geometry.left}" x2="${geometry.width - geometry.right}" y1="${geometry.height - geometry.bottom}" y2="${geometry.height - geometry.bottom}"/>
            <text class="protein-axis-label" x="6" y="${geometry.top + 5}">${escapeHtml(formatNumber(geometry.maxValue))}</text>
            <text class="protein-axis-label" x="6" y="${geometry.height - geometry.bottom}">${escapeHtml(formatNumber(geometry.minValue))}</text>
            <text class="protein-axis-label" x="${geometry.left}" y="${geometry.height - 8}">${geometry.minCycle}</text>
            <text class="protein-axis-label" x="${geometry.width - geometry.right}" y="${geometry.height - 8}" text-anchor="end">${geometry.maxCycle}</text>
            ${renderSeries(series.cycleBest, geometry, 'cycle-best')}
            ${renderSeries(series.globalBest, geometry, 'global-best')}
            <line class="protein-chart-crosshair" y1="18" y2="${geometry.height - geometry.bottom}" visibility="hidden"/>
          </svg>
          <div class="protein-chart-tooltip" hidden></div>
        </div>`;
  }

  function renderMetricChart(metric, series, active = false) {
    const plot = renderMetricPlot(metric, series, { width: 560, height: 220 });
    if (!plot) return '';
    return `
      <article class="protein-metric-card" data-metric="${escapeHtml(metric)}">
        <header><h3>${escapeHtml(humanizeFieldName(metric))}</h3><div class="protein-chart-legend protein-chart-legend-compact"><span class="cycle-best">Current cycle best</span><span class="global-best">Global best</span></div></header>
        ${plot}
      </article>`;
  }

  function buildTreeLayout(tree, options = {}) {
    const width = options.width || 1000;
    const rowHeight = options.rowHeight || 76;
    const sourceNodes = Array.isArray(tree?.nodes) ? tree.nodes.slice(0, 2000) : [];
    const rows = new Map();
    for (const node of sourceNodes) {
      const level = Number.isFinite(node.cycle) ? node.cycle : -1;
      if (!rows.has(level)) rows.set(level, []);
      rows.get(level).push(node);
    }
    const nodes = [];
    const levels = [...rows.keys()].sort((a,b) => a-b);
    for (const [rowIndex, level] of levels.entries()) {
      const row = rows.get(level);
      row.sort((a, b) => (a.cycle ?? -1) - (b.cycle ?? -1) || String(a.id).localeCompare(String(b.id)));
      row.forEach((node, index) => {
        nodes.push({
          ...node,
          x: 90 + ((index + 1) / (row.length + 1)) * (width - 114),
          y: 34 + rowIndex * rowHeight
        });
      });
    }
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const edges = (tree?.edges || [])
      .map((edge) => ({ source: byId.get(edge.source), target: byId.get(edge.target) }))
      .filter((edge) => edge.source && edge.target);
    return { width, height: Math.max(220, 80 + Math.max(0, levels.length - 1) * rowHeight), nodes, edges, levels };
  }

  function renderTree(tree) {
    const layout = buildTreeLayout(tree);
    const edges = layout.edges
      .map(
        ({ source, target }) =>
          `<path data-source="${escapeHtml(source.id)}" data-target="${escapeHtml(target.id)}" d="M ${source.x} ${source.y} C ${source.x} ${(source.y + target.y) / 2}, ${target.x} ${(source.y + target.y) / 2}, ${target.x} ${target.y}"/>`
      )
      .join('');
    const nodes = layout.nodes
      .map((node) => {
        const candidateLabel = node.candidateId || node.label || node.id;
        const detail = node.kind === 'candidate'
          ? `${candidateLabel} · ${formatNumber(node.objective)}${node.skillId ? ` · ${node.skillId}` : ''}`
          : candidateLabel;
        const radius = node.kind === 'cycle' ? 7 : node.kind === 'initial' ? 8 : 5;
        return `<g class="protein-tree-node protein-tree-node-${escapeHtml(node.kind)}" data-node-id="${escapeHtml(node.id)}" data-detail="${escapeHtml(`Cycle ${node.cycle ?? '—'} · ${detail}`)}" transform="translate(${node.x} ${node.y})" tabindex="0" aria-label="${escapeHtml(detail)}"><circle r="${radius}"/></g>`;
      })
      .join('');
    return `
      <div class="protein-tree-toolbar">
        <button type="button" data-tree-action="zoom-in">+</button>
        <button type="button" data-tree-action="zoom-out">−</button>
        <button type="button" data-tree-action="reset">Reset</button>
        <div class="protein-tree-legend" aria-label="Tree legend"><span class="is-cycle">Cycle</span><span class="is-candidate">Candidate</span><span class="is-selected">Selected</span></div>
      </div>
      <div class="protein-tree-scroll"><svg class="protein-tree-svg" style="height:${layout.height}px" viewBox="0 0 ${layout.width} ${layout.height}" preserveAspectRatio="none" role="img" aria-label="Candidate search tree">
        <g class="protein-tree-viewport"><g class="protein-tree-cycle-axis">${layout.levels.map((cycle,index) => `<text x="8" y="${38+index*76}">${cycle < 0 ? 'Initial' : 'Cycle '+cycle}</text>`).join('')}</g><g class="protein-tree-edges">${edges}</g><g class="protein-tree-nodes">${nodes}</g></g>
      </svg></div><div class="protein-tree-detail">Click a node to keep its candidate details visible.</div>`;
  }

  function activityEventId(event) {
    return event.event_id || `${event.timestamp || ''}:${event.event_type || ''}:${event.status || ''}`;
  }

  function renderActivityBody(event) {
    const eventId = activityEventId(event);
    const outputEmptyLabel = event.status === 'started' ? 'Pending…' : 'No output captured';
    const metadata = event.metadata && Object.keys(event.metadata).length
      ? `<section class="protein-activity-metadata"><h4>Metadata</h4>${renderPayloadValue(event.metadata)}</section>`
      : '';
    const error = event.error
      ? `<section class="protein-activity-error"><h4>Error</h4>${renderPayloadValue(event.error)}</section>`
      : '';
    return `<div class="protein-activity-summary">${escapeHtml(event.summary || '')}</div>
      <div class="protein-activity-io-grid">
        ${renderPayloadPanel('Input', event.input_payload, eventId, 'No input captured')}
        ${renderPayloadPanel('Output', event.output_payload, eventId, outputEmptyLabel)}
      </div>
      ${metadata}${error}`;
  }

  function renderActivity(events, options = {}) {
    const deferPayloads = options.deferPayloads === true;
    const rows = (events || []).slice().reverse().map((event) => {
      const duration = typeof event.duration_ms === 'number' ? `${Math.round(event.duration_ms)} ms` : '-';
      const cycle = event.cycle == null ? 'Setup' : `Cycle ${event.cycle}`;
      const subject = event.skill || event.tool || event.actor || event.phase || event.event_type;
      const eventId = activityEventId(event);
      const hasInput = event.input_payload !== null && event.input_payload !== undefined;
      const hasOutput = event.output_payload !== null && event.output_payload !== undefined;
      const body = deferPayloads
        ? '<div class="protein-activity-lazy">Expand to load full Input / Output.</div>'
        : renderActivityBody(event);
      return `<details class="protein-activity-event protein-status-${escapeHtml(event.status)}" data-event-id="${escapeHtml(eventId)}" data-persist-key="activity:${escapeHtml(eventId)}">
        <summary>
          <span class="protein-activity-cycle">${escapeHtml(cycle)}</span>
          <span class="protein-activity-phase">${escapeHtml(event.phase || event.event_type || '-')}</span>
          <strong>${escapeHtml(subject || '-')}</strong>
          <span>${escapeHtml(event.status || '-')}</span>
          <span>${escapeHtml(duration)}</span>
          <span>${event.candidate_count == null ? '-' : `${event.candidate_count} candidates`}</span>
          <time datetime="${escapeHtml(event.timestamp || '')}">${escapeHtml(formatTimestamp(event.timestamp))}</time>
          <span class="protein-activity-io"><span class="${hasInput ? 'is-captured' : ''}">Input</span><span class="${hasOutput ? 'is-captured' : ''}">Output</span></span>
        </summary>
        <div class="protein-activity-detail-body" data-activity-body="${escapeHtml(eventId)}">${body}</div>
      </details>`;
    }).join('');
    return rows || '<div class="protein-empty-inline">Waiting for the first design event…</div>';
  }

  const DESIGN_LOOP_STAGES = [
    ['design', 'Design'],
    ['fold', 'Fold'],
    ['reflection', 'Reflection'],
    ['parent_selection', 'Parent selection']
  ];

  function normalizeDesignPhase(phase) {
    if (phase === 'setup' || phase === 'analysis' || phase === 'quality') return 'design';
    if (phase === 'population' || phase === 'population_update' || phase === 'parent_selection') return 'parent_selection';
    if (phase === 'post_refold') return 'fold';
    if (phase === 'post_filter' || phase === 'final_visualization') return 'reflection';
    return phase;
  }

  function renderDesignLoop(phase, runStatus, cycle, totalCycles) {
    const normalized = normalizeDesignPhase(phase);
    const currentIndex = DESIGN_LOOP_STAGES.findIndex(([name]) => name === normalized);
    const stages = DESIGN_LOOP_STAGES.map(([name, label], index) => {
      let state = 'pending';
      if (runStatus === 'completed') state = 'completed';
      else if (index < currentIndex) state = 'completed';
      else if (index === currentIndex) state = 'current';
      return `<li class="protein-loop-stage protein-loop-stage-${name} is-${state}${state === 'current' && runStatus === 'running' ? ' is-executing' : ''}" data-loop-stage="${name}" role="button" tabindex="0" aria-pressed="false" aria-current="${state === 'current' ? 'step' : 'false'}">
        <span class="protein-loop-stage-index" aria-hidden="true">${index + 1}</span>
        <span class="protein-loop-stage-copy"><strong>${escapeHtml(label)}</strong><small>${state === 'current' ? 'In progress' : state}</small></span>
      </li>`;
    }).join('');
    return `<div class="protein-design-loop" role="group" aria-label="Design, fold, reflection, and parent selection loop">
      <svg class="protein-loop-connectors" viewBox="0 0 760 286" aria-hidden="true">
        <defs><marker id="loop-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="currentColor" stroke-width="1.5"/></marker></defs>
        <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#loop-arrowhead)">
          <path d="M 467 46 C 570 46 653 55 653 108"/>
          <path d="M 653 174 C 653 240 560 240 478 240"/>
          <path d="M 293 240 C 190 240 107 231 107 181"/>
          <path d="M 107 114 C 107 46 200 46 282 46"/>
        </g>
      </svg>
      <ol>${stages}</ol>
      <div class="protein-loop-center"><span>Current search</span><strong>Cycle ${escapeHtml(cycle || 0)}</strong><small>of ${escapeHtml(totalCycles || 0)}</small></div>
    </div>`;
  }


  function renderPostFilter(postFilter) {
    if (!postFilter) {
      return '<div class="protein-empty-inline">No post-filter stage data for this run.</div>';
    }
    const errors = [postFilter.postRefoldError, postFilter.postFilterError].filter(Boolean);
    const decisions = (postFilter.decisions || []).map((decision) => {
      const metricRows = Object.entries(decision.metrics || {})
        .filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, value]) => `<div><dt>${escapeHtml(humanizeFieldName(name))}</dt><dd>${escapeHtml(formatNumber(value))}</dd></div>`)
        .join('');
      const structurePanel = decision.structureArtifactPath
        ? `<section class="protein-filter-structure-panel">
            <header><strong>Refolded final structure</strong><span>${escapeHtml(decision.candidateId)}</span></header>
            <div class="protein-filter-structure" data-post-filter-structure
              data-structure-path="${escapeHtml(decision.structureArtifactPath)}"
              data-target-chain-ids="${escapeHtml(JSON.stringify(decision.targetChainIds || []))}"
              data-binder-chain-ids="${escapeHtml(JSON.stringify(decision.binderChainIds || []))}">
              <div class="protein-structure-message">Structure will load when visible.</div>
            </div>
          </section>`
        : `<section class="protein-filter-structure-panel">
            <header><strong>Refolded final structure</strong><span>${escapeHtml(decision.candidateId)}</span></header>
            <div class="protein-filter-structure protein-filter-structure-missing"><div class="protein-structure-message">No captured structure for this candidate.</div></div>
          </section>`;
      return `
      <article class="protein-filter-decision${decision.passFilter ? ' is-selected' : ''}" data-candidate-id="${escapeHtml(decision.candidateId)}">
        <div class="protein-filter-decision-head">
          <div class="protein-filter-rank">#${escapeHtml(decision.rank || '-')}</div>
          <div class="protein-filter-decision-main">
            <div><strong>${escapeHtml(decision.candidateId)}</strong><span>${decision.passFilter ? 'Selected' : decision.hardEligible ? 'Not selected' : 'Hard rejected'}</span></div>
            <p>${escapeHtml(decision.rationale || 'No rationale captured.')}</p>
            ${decision.strengths?.length ? `<small>Strengths · ${escapeHtml(decision.strengths.join('; '))}</small>` : ''}
            ${decision.risks?.length ? `<small class="protein-filter-risk">Risks · ${escapeHtml(decision.risks.join('; '))}</small>` : ''}
          </div>
        </div>
        <dl class="protein-filter-metrics"><div><dt>Objective</dt><dd>${escapeHtml(formatNumber(decision.objective))}</dd></div>${metricRows}</dl>
        ${structurePanel}
      </article>`;
    }).join('');
    return `<div class="protein-filter-overview">
        <div class="protein-filter-summary">
          <div><span>Status</span><strong>${escapeHtml(humanizeFieldName(postFilter.status))}</strong></div>
          <div><span>Selection mode</span><strong>${escapeHtml(humanizeFieldName(postFilter.mode))}</strong></div>
          <div><span>Successfully refolded</span><strong>${escapeHtml(postFilter.refoldedCandidateCount)}</strong></div>
          <div><span>Eligible</span><strong>${escapeHtml(postFilter.eligibleCandidateCount)}</strong></div>
          <div><span>Selected</span><strong>${escapeHtml(postFilter.selectedCandidateIds?.length || 0)} / ${escapeHtml(postFilter.topK || '-')}</strong></div>
        </div>
        <div class="protein-filter-strategy"><strong>Selection criteria</strong><p>${escapeHtml(postFilter.strategySummary || '-')}</p></div>
      </div>
      ${errors.length ? `<div class="protein-filter-errors">${errors.map((error) => `<p>${escapeHtml(error)}</p>`).join('')}</div>` : ''}
      <div class="protein-filter-decisions">${decisions || '<div class="protein-empty-inline">Post-filter ran without candidate decisions.</div>'}</div>`;
  }

  function renderProteinDesignDashboard(payload) {
    const runs = payload?.runs || [];
    const run = payload?.run || null;
    if (payload?.error) {
      return `<div class="protein-empty protein-error"><h2>Protein-design data unavailable</h2><p>${escapeHtml(payload.error)}</p></div>`;
    }
    if (!runs.length) {
      return `<div class="protein-empty"><h2>No protein-design runs</h2><p>Start a design task from OpenDDE Harness to populate this workspace.</p></div>`;
    }
    const selectedRunId = payload.selectedRunId || run?.taskId || runs[0].taskId;
    const runPicker = renderRunPicker(runs, selectedRunId);
    if (!run) {
      return `<div class="protein-dashboard"><header class="protein-runbar">${runPicker}</header><div class="protein-loading">Loading selected run…</div></div>`;
    }
    const availableMetrics = orderMetricNames(run.metricNames, run.objectiveKey)
      .filter((metric) => !/(^|[._])ptm$/i.test(metric) && chartGeometry(run.series?.[metric]))
      .slice(0, 8);
    const metricCharts = availableMetrics
      .map((metric, index) => renderMetricChart(metric, run.series?.[metric], index === 0))
      .join('');
    const traceUnavailable = !run.traceId || run.traceAvailable === false;
    return `<div id="proteinCandidateWorkspace">${candidatesDashboard.renderProteinDesignDashboard(payload).replace('<!--post-filter-results-->', renderPostFilter(run.postFilter))}</div>
      <div class="protein-design-details">
        <div class="protein-process-grid">
        <section class="protein-section protein-loop-section">
          <header><h2>Design loop</h2><span>Cycle ${escapeHtml(run.cycle || 0)} of ${escapeHtml(run.totalCycles || 0)}</span><button class="ghost-button" id="proteinOpenTrace" type="button"${traceUnavailable ? ' disabled title="Trace is no longer available"' : ''}>Open trace</button></header>
          ${renderDesignLoop(run.phase || run.events?.at(-1)?.phase, run.status, run.cycle, run.totalCycles)}
          <div class="protein-loop-io">
            <div class="protein-loop-io-toolbar" hidden><strong data-loop-selection></strong><label>Cycle <select data-loop-cycle><option value="all">All cycles</option>${[...new Set((run.events || []).map(event => event.cycle).filter(cycle => cycle != null))].sort((a,b) => b-a).map(cycle => `<option value="${escapeHtml(cycle)}">${escapeHtml(cycle)}</option>`).join('')}</select></label></div>
            <div class="protein-activity-list"><div class="protein-empty-inline">Select a step above to inspect its Input / Output.</div></div>
          </div>
        </section>
        <section class="protein-section protein-tree-section">
          <header><h2>Search tree</h2><span>${run.tree?.nodes?.length || 0} nodes</span></header>
          <div class="protein-tree-stage">${renderTree(run.tree)}</div>
        </section>
        </div>
        <section class="protein-section protein-metrics-section">
          <header><h2>Metric trends</h2><span>Current cycle best vs. global best</span></header>
          ${metricCharts ? `<div class="protein-metric-explorer">
            <div class="protein-metric-grid" aria-label="Metric trends">${metricCharts}</div>
          </div>` : '<div class="protein-empty-inline">No numeric metrics yet.</div>'}
        </section>
      </div>`;
  }

  function load3Dmol() {
    if (typeof window === 'undefined') return Promise.reject(new Error('3Dmol requires a browser'));
    if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
    if (threeDmolPromise) return threeDmolPromise;
    threeDmolPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = '/vendor/3Dmol-min.js';
      script.async = true;
      script.onload = () => {
        if (window.$3Dmol) {
          resolve(window.$3Dmol);
          return;
        }
        threeDmolPromise = null;
        reject(new Error('Local 3Dmol.js loaded without exposing its viewer API'));
      };
      script.onerror = () => {
        threeDmolPromise = null;
        reject(new Error('Failed to load the bundled 3Dmol.js viewer'));
      };
      document.head.appendChild(script);
    });
    return threeDmolPromise;
  }

  function parseChainIds(value) {
    if (Array.isArray(value)) return value.map(String);
    try {
      const parsed = JSON.parse(value || '[]');
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch {
      return [];
    }
  }

  async function createStructureViewer(stage, artifactPath, fallbackTargetChains, fallbackBinderChains) {
    const [threeDmol, response] = await Promise.all([
      load3Dmol(),
      fetch(`/api/artifact?path=${encodeURIComponent(artifactPath)}`, { cache: 'no-store' })
    ]);
    if (!response.ok) throw new Error(`Structure request failed (${response.status})`);
    const artifact = await response.json();
    const structure = artifact.parsed || JSON.parse(artifact.content || '{}');
    if (!structure.text || !structure.format) throw new Error('Structure artifact is incomplete');
    const targetChainIds = Array.isArray(structure.target_chain_ids)
      ? structure.target_chain_ids.map(String)
      : parseChainIds(fallbackTargetChains);
    const binderChainIds = Array.isArray(structure.binder_chain_ids)
      ? structure.binder_chain_ids.map(String)
      : parseChainIds(fallbackBinderChains);
    stage.innerHTML = '';
    const viewer = threeDmol.createViewer(stage, { backgroundColor: document.documentElement.dataset.theme === 'dark' ? '#191c24' : '#ffffff', antialias: true });
    viewer.addModel(structure.text, structure.format);
    return { viewer, targetChainIds, binderChainIds };
  }

  function bindChartTooltips(rootElement) {
    const charts = [...rootElement.querySelectorAll('.protein-chart-wrap')].map(wrap => ({
      wrap, tooltip: wrap.querySelector('.protein-chart-tooltip'),
      crosshair: wrap.querySelector('.protein-chart-crosshair'),
      points: [...wrap.querySelectorAll('circle[data-cycle]')]
    }));
    const show = cycle => charts.forEach(({tooltip, crosshair, points}) => {
      const matches = points.filter(point => point.dataset.cycle === cycle);
      points.forEach(point => point.classList.toggle('is-hovered', matches.includes(point)));
      tooltip.hidden = cycle == null;
      crosshair.setAttribute('visibility', matches.length ? 'visible' : 'hidden');
      if (cycle == null) return;
      if (matches.length) {
        crosshair.setAttribute('x1', matches[0].getAttribute('cx'));
        crosshair.setAttribute('x2', matches[0].getAttribute('cx'));
      }
      const value = name => {
        const point = matches.find(point => point.classList.contains(name + '-point'));
        return point ? formatNumber(Number(point.dataset.value)) : '—';
      };
      tooltip.textContent = `Cycle ${cycle}\nCurrent cycle best: ${value('cycle-best')}\nGlobal best: ${value('global-best')}`;
    });
    charts.forEach(({points}) => points.forEach(point => {
      point.addEventListener('pointerenter', () => show(point.dataset.cycle));
      point.addEventListener('focus', () => show(point.dataset.cycle));
      point.addEventListener('pointerleave', () => show(null));
      point.addEventListener('blur', () => show(null));
    }));
  }

  async function showPostFilterStructure(stage) {
    if (!stage || stage.dataset.loadState) return;
    stage.dataset.loadState = 'loading';
    stage.innerHTML = '<div class="protein-structure-message">Loading post-filter structure…</div>';
    try {
      const { viewer, targetChainIds, binderChainIds } = await createStructureViewer(
        stage,
        stage.dataset.structurePath,
        stage.dataset.targetChainIds,
        stage.dataset.binderChainIds
      );
      viewer.setStyle({}, { cartoon: { color: '#cccccc' } });
      targetChainIds.forEach((chain) => {
        viewer.setStyle({ chain }, { cartoon: { color: '#cccccc' } });
      });
      binderChainIds.forEach((chain) => {
        viewer.setStyle({ chain }, { cartoon: { color: '#a194f3' } });
      });
      viewer.zoomTo();
      viewer.render();
      viewer.resize();
      stage._proteinViewer = viewer;
      stage.dataset.loadState = 'loaded';
    } catch (error) {
      stage.dataset.loadState = 'failed';
      stage.innerHTML = `<div class="protein-structure-message protein-error">${escapeHtml(error.message)}</div>`;
    }
  }

  function bindPostFilterStructures(rootElement) {
    rootElement._proteinPostFilterObserver?.disconnect();
    const stages = [...rootElement.querySelectorAll('[data-post-filter-structure]')];
    if (!stages.length) return;
    if (typeof IntersectionObserver !== 'function') {
      stages.forEach((stage) => showPostFilterStructure(stage));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        showPostFilterStructure(entry.target);
      });
    }, { rootMargin: '240px 0px' });
    stages.forEach((stage) => observer.observe(stage));
    rootElement._proteinPostFilterObserver = observer;
  }

  function bindTree(rootElement, run) {
    const svg = rootElement.querySelector('.protein-tree-svg');
    const viewport = svg?.querySelector('.protein-tree-viewport');
    if (!svg || !viewport) return;
    const nodes = [...svg.querySelectorAll('[data-node-id]')];
    const edges = [...svg.querySelectorAll('[data-source]')];
    const detail = rootElement.querySelector('.protein-tree-detail');
    const previewState = rootElement._treePreview || {element: document.createElement('div'), pinned: false};
    rootElement._treePreview = previewState;
    const preview = previewState.element;
    preview.className = `protein-node-preview${previewState.pinned ? ' is-pinned' : ''}`;
    preview.hidden = !previewState.pinned;
    const treeScroll = rootElement.querySelector('.protein-tree-scroll');
    treeScroll.appendChild(preview);
    const positionPreview = node => {
      if (!node) return;
      const bounds = node.getBoundingClientRect(), container = treeScroll.getBoundingClientRect();
      const gap = 10;
      const leftEdge = Math.max(container.left, 0) + gap;
      const rightEdge = Math.min(container.right, window.innerWidth) - gap;
      const topEdge = Math.max(container.top, 54) + gap;
      const bottomEdge = Math.min(container.bottom, window.innerHeight) - gap;
      preview.style.maxWidth = `${Math.max(120, rightEdge - leftEdge)}px`;
      preview.style.maxHeight = `${Math.max(120, bottomEdge - topEdge)}px`;
      const width = preview.offsetWidth, height = preview.offsetHeight;
      const preferredLeft = bounds.right + gap + width <= rightEdge ? bounds.right + gap : bounds.left - width - gap;
      const preferredTop = bounds.top + height <= bottomEdge ? bounds.top : bounds.bottom - height;
      const left = Math.max(leftEdge, Math.min(preferredLeft, rightEdge - width));
      const top = Math.max(topEdge, Math.min(preferredTop, bottomEdge - height));
      preview.style.left = `${left - container.left + treeScroll.scrollLeft}px`;
      preview.style.top = `${top - container.top + treeScroll.scrollTop}px`;
    };
    if (previewState.pinned) positionPreview(nodes.find(node => node.dataset.nodeId === previewState.id));
    let previewTimer;
    const hidePreview = () => { if (previewState.pinned) return; clearTimeout(previewTimer); preview.hidden = true; previewState.viewer?.clear(); };
    const showPreview = (node, pin = false) => {
      if (previewState.pinned && !pin) return;
      if (pin && previewState.id === node.dataset.nodeId && !preview.hidden) {
        previewState.pinned = true; preview.classList.add('is-pinned'); return;
      }
      previewState.pinned = false;
      hidePreview();
      const id = node.dataset.nodeId.replace(/^candidate:/, '');
      const candidate = candidatesDashboard.allCandidates(run).find(item => item.candidateId === id);
      if (!candidate) return;
      previewState.pinned = pin;
      previewState.id = node.dataset.nodeId;
      preview.classList.toggle('is-pinned', pin);
      previewTimer = setTimeout(async () => {
        const metrics = [[humanizeFieldName(run.objectiveKey), candidate.objective], ['ipTM', candidate.metrics?.iptm], ['Ranking score', candidate.metrics?.ranking_score], ['pLDDT', candidate.metrics?.plddt]];
        preview.innerHTML = `<header><div><small>Cycle ${escapeHtml(candidate.cycle)} · Candidate</small><strong>${escapeHtml(id)}</strong></div><button type="button" aria-label="Close candidate preview">×</button></header><div class="protein-node-preview-structure">Loading structure…</div><div class="protein-node-preview-metrics">${metrics.map(([label,value]) => `<div><small>${escapeHtml(label)}</small><strong>${escapeHtml(formatNumber(value))}</strong></div>`).join('')}</div><section class="protein-node-sequence"><strong>Sequence · ${candidate.sequence?.length || 0} residues</strong>${candidatesDashboard.renderFullCandidateSequence(candidate.sequence || '', candidatesDashboard.candidateCdrRegions(candidate, run))}</section><footer>Click node to pin · Drag to rotate · Scroll to zoom</footer>`;
        preview.querySelector('button').onclick = () => { previewState.pinned = false; hidePreview(); };
        preview.hidden = false;
        positionPreview(node);
        const stage = preview.querySelector('.protein-node-preview-structure');
        const artifact = (run.structures || []).find(item => item.candidateId === id);
        if (!artifact) { stage.textContent = 'Structure unavailable'; return; }
        try {
          const {viewer, targetChainIds, binderChainIds} = await createStructureViewer(stage, artifact.artifactPath, artifact.targetChainIds, artifact.binderChainIds);
          if (!stage.isConnected || preview.hidden) { viewer.clear(); return; }
          previewState.viewer = viewer;
          viewer.setStyle({}, {cartoon:{color:'#cccccc'}});
          binderChainIds.forEach(chain => viewer.setStyle({chain}, {cartoon:{color:'#a194f3'}}));
          viewer.zoomTo(); viewer.render();
          if (targetChainIds.length) {
            const target = {chain: targetChainIds.filter(chain => !binderChainIds.includes(chain))};
            if (target.chain.length) {
              viewer.setStyle(target, {});
              await viewer.addSurface(window.$3Dmol.SurfaceType.MS, {color:'#dfe2e5', opacity:0.5}, target);
              if (!stage.isConnected || preview.hidden) { viewer.clear(); return; }
              viewer.render();
            }
          }
        } catch { if (stage.isConnected) stage.textContent = 'Structure unavailable'; }
      }, pin ? 0 : 250);
    };
    const highlight = node => {
      const ancestors = new Set(node ? [node.dataset.nodeId] : []);
      let changed = true;
      while (changed) {
        changed = false;
        edges.forEach(edge => {
          if (ancestors.has(edge.dataset.target) && !ancestors.has(edge.dataset.source)) {
            ancestors.add(edge.dataset.source); changed = true;
          }
        });
      }
      nodes.forEach(item => item.classList.toggle('is-highlighted', ancestors.has(item.dataset.nodeId)));
      edges.forEach(edge => edge.classList.toggle('is-highlighted', ancestors.has(edge.dataset.target)));
      if (detail) detail.textContent = node?.dataset.detail || 'Hover a node to inspect its lineage and candidate details.';
    };
    let selectedNode = nodes.find(node => node.dataset.nodeId === rootElement._selectedTreeNode);
    highlight(selectedNode);
    nodes.forEach(node => {
      node.addEventListener('pointerenter', () => highlight(node));
      node.addEventListener('pointerenter', () => showPreview(node));
      node.addEventListener('pointerleave', hidePreview);
      node.addEventListener('focus', () => highlight(node));
      node.addEventListener('pointerleave', () => highlight(selectedNode));
      node.addEventListener('blur', () => highlight(selectedNode));
      const select = () => { selectedNode = node; rootElement._selectedTreeNode = node.dataset.nodeId; highlight(node); showPreview(node, true); };
      node.addEventListener('click', select);
      node.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
    });
    let scale = 1;
    let x = 0;
    let y = 0;
    let drag = null;
    const update = () => {
      viewport.setAttribute('transform', `translate(${x} ${y}) scale(${scale})`);
      if (previewState.pinned) positionPreview(nodes.find(node => node.dataset.nodeId === previewState.id));
    };
    const zoom = (factor) => {
      scale = Math.min(3, Math.max(0.35, scale * factor));
      update();
    };
    svg.addEventListener('wheel', (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoom(event.deltaY < 0 ? 1.1 : 0.9);
    }, { passive: false });
    svg.addEventListener('pointerdown', (event) => {
      drag = { clientX: event.clientX, clientY: event.clientY, x, y };
      if (!event.target.closest('[data-node-id]')) svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener('pointermove', (event) => {
      if (!drag) return;
      x = drag.x + event.clientX - drag.clientX;
      y = drag.y + event.clientY - drag.clientY;
      update();
    });
    svg.addEventListener('pointerup', () => { drag = null; });
    rootElement.querySelectorAll('[data-tree-action]').forEach((button) => {
      button.addEventListener('click', () => {
        if (button.dataset.treeAction === 'zoom-in') zoom(1.2);
        if (button.dataset.treeAction === 'zoom-out') zoom(0.8);
        if (button.dataset.treeAction === 'reset') {
          scale = 1; x = 0; y = 0; update();
        }
      });
    });
    rootElement.querySelectorAll('.protein-tree-node-candidate').forEach((node) => {
      node.addEventListener('click', () => {
        rootElement.querySelectorAll('.protein-tree-node').forEach((item) => item.classList.toggle('is-selected', item === node));
        const candidateId = node.dataset.nodeId?.replace(/^candidate:/, '');
        const structureButton = [...rootElement.querySelectorAll('.protein-structure-option')]
          .find((item) => item.dataset.candidateId === candidateId);
        if (structureButton) showStructure(rootElement, structureButton);
      });
    });
  }

  function bindLazyActivity(rootElement, events) {
    const byId = new Map((events || []).map((event) => [activityEventId(event), event]));
    const hydrate = (details) => {
      if (!details?.open || details.dataset.activityHydrated === 'true') return;
      const event = byId.get(details.dataset.eventId);
      const body = details.querySelector('[data-activity-body]');
      if (!event || !body) return;
      body.innerHTML = renderActivityBody(event);
      details.dataset.activityHydrated = 'true';
    };
    rootElement.querySelectorAll('.protein-activity-event').forEach((details) => {
      details.addEventListener('toggle', () => hydrate(details));
      hydrate(details);
    });
    return () => rootElement.querySelectorAll('.protein-activity-event[open]').forEach(hydrate);
  }

  function applyStructureFilter(rootElement, value) {
    const query = String(value || '').trim().toLowerCase();
    rootElement.querySelectorAll('.protein-structure-option').forEach((button) => {
      const searchable = `${button.dataset.candidateId || ''} cycle ${button.dataset.cycle || ''}`.toLowerCase();
      button.hidden = Boolean(query) && !searchable.includes(query);
    });
  }

  function captureDashboardViewState(rootElement) {
    const structureList = rootElement.querySelector('.protein-structure-list');
    const filter = rootElement.querySelector('#proteinStructureFilter');
    const style = rootElement.querySelector('#proteinStructureStyle');
    const structureSection = rootElement.querySelector('.protein-structure-section');
    const spinButton = rootElement.querySelector('[data-structure-action="spin"]');
    const treeViewport = rootElement.querySelector('.protein-tree-viewport');
    const runPicker = rootElement.querySelector('#proteinDesignRunPicker');
    const runPickerMenu = rootElement.querySelector('.protein-run-picker-menu');
    const activityList = rootElement.querySelector('.protein-activity-list');
    const openDetails = [...rootElement.querySelectorAll('details[data-persist-key][open]')]
      .map((details) => details.dataset.persistKey)
      .filter(Boolean);
    const activityPayloadScroll = Object.fromEntries(
      [...rootElement.querySelectorAll('[data-activity-scroll-key]')]
        .map((element) => [element.dataset.activityScrollKey, element.scrollTop || 0])
    );
    return {
      sceneScrollTop: rootElement.scrollTop || 0,
      sceneScrollLeft: rootElement.scrollLeft || 0,
      structureScrollTop: structureList?.scrollTop || 0,
      structureScrollLeft: structureList?.scrollLeft || 0,
      structureFilter: filter?.value || '',
      structureStyle: style?.value || 'cartoon',
      structureExpanded: Boolean(structureSection?.classList.contains('is-expanded')),
      structureSpinning: Boolean(spinButton?.classList.contains('is-active')),
      treeTransform: treeViewport?.getAttribute('transform') || '',
      treeScrollTop: rootElement.querySelector('.protein-tree-scroll')?.scrollTop || 0,
      runPickerOpen: Boolean(runPicker?.open),
      runPickerScrollTop: runPickerMenu?.scrollTop || 0,
      activityScrollTop: activityList?.scrollTop || 0,
      activityScrollLeft: activityList?.scrollLeft || 0,
      selectedMetric: rootElement.querySelector('[data-metric-select].is-active')?.dataset.metric || '',
      openDetails,
      activityPayloadScroll
    };
  }

  function restoreDashboardViewState(rootElement, viewState) {
    if (!viewState) return;
    const filter = rootElement.querySelector('#proteinStructureFilter');
    if (filter) {
      filter.value = viewState.structureFilter;
      applyStructureFilter(rootElement, viewState.structureFilter);
    }
    const style = rootElement.querySelector('#proteinStructureStyle');
    if (style) style.value = viewState.structureStyle;
    const structureSection = rootElement.querySelector('.protein-structure-section');
    const expandButton = rootElement.querySelector('[data-structure-action="expand"]');
    if (viewState.structureExpanded) {
      structureSection?.classList.add('is-expanded');
      if (expandButton) expandButton.textContent = 'Close';
    }
    const spinButton = rootElement.querySelector('[data-structure-action="spin"]');
    spinButton?.classList.toggle('is-active', viewState.structureSpinning);
    const treeViewport = rootElement.querySelector('.protein-tree-viewport');
    if (treeViewport && viewState.treeTransform) treeViewport.setAttribute('transform', viewState.treeTransform);
    const treeScroll = rootElement.querySelector('.protein-tree-scroll');
    if (treeScroll) treeScroll.scrollTop = viewState.treeScrollTop || 0;
    const runPicker = rootElement.querySelector('#proteinDesignRunPicker');
    if (runPicker) runPicker.open = viewState.runPickerOpen;
    const openDetails = new Set(viewState.openDetails || []);
    rootElement.querySelectorAll('details[data-persist-key]').forEach((details) => {
      details.open = openDetails.has(details.dataset.persistKey);
    });

    const restoreScroll = () => {
      rootElement.scrollTop = viewState.sceneScrollTop;
      rootElement.scrollLeft = viewState.sceneScrollLeft;
      const structureList = rootElement.querySelector('.protein-structure-list');
      if (structureList) {
        structureList.scrollTop = viewState.structureScrollTop;
        structureList.scrollLeft = viewState.structureScrollLeft;
      }
      const runPickerMenu = rootElement.querySelector('.protein-run-picker-menu');
      if (runPickerMenu) runPickerMenu.scrollTop = viewState.runPickerScrollTop;
      const activityList = rootElement.querySelector('.protein-activity-list');
      if (activityList) {
        activityList.scrollTop = viewState.activityScrollTop;
        activityList.scrollLeft = viewState.activityScrollLeft;
      }
      rootElement.querySelectorAll('[data-activity-scroll-key]').forEach((element) => {
        element.scrollTop = viewState.activityPayloadScroll?.[element.dataset.activityScrollKey] || 0;
      });
    };
    restoreScroll();
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(restoreScroll);
  }

  function bindDesignLoop(rootElement, run, sameRun) {
    if (!sameRun) { rootElement._loopPhase = null; rootElement._loopCycle = 'all'; rootElement._loopFollow = true; }
    const activePhase = rootElement.querySelector('[data-loop-stage].is-current')?.dataset.loopStage;
    if (rootElement._loopFollow !== false && activePhase) rootElement._loopPhase = activePhase;
    const select = rootElement.querySelector('[data-loop-cycle]');
    let hydrate = () => {};
    if (!select) return hydrate;
    const follow = document.createElement('button');
    follow.type = 'button';
    follow.className = 'ghost-button';
    rootElement.querySelector('.protein-loop-io-toolbar').appendChild(follow);
    const update = () => {
      follow.textContent = rootElement._loopFollow !== false ? 'Following live' : 'Follow live';
      follow.setAttribute('aria-pressed', String(rootElement._loopFollow !== false));
      const phase = rootElement._loopPhase;
      rootElement.querySelectorAll('[data-loop-stage]').forEach(node => node.setAttribute('aria-pressed', String(node.dataset.loopStage === phase)));
      if (!phase) return;
      rootElement.querySelector('.protein-loop-io-toolbar').hidden = false;
      rootElement.querySelector('[data-loop-selection]').textContent = DESIGN_LOOP_STAGES.find(([key]) => key === phase)?.[1] || phase;
      const events = (run?.events || []).filter(event => normalizeDesignPhase(event.phase) === phase && (select.value === 'all' || String(event.cycle) === select.value));
      rootElement.querySelector('.protein-activity-list').innerHTML = renderActivity(events, {deferPayloads:true});
      hydrate = bindLazyActivity(rootElement, events);
      hydrate();
    };
    select.value = [...select.options].some(option => option.value === rootElement._loopCycle) ? rootElement._loopCycle : 'all';
    select.hidden = true;
    const picker = document.createElement('details');
    picker.className = 'protein-cycle-picker';
    picker.innerHTML = `<summary>${escapeHtml(select.selectedOptions[0]?.textContent || 'All cycles')} <span>⌄</span></summary><div>${[...select.options].map(option => `<button type="button" data-cycle-value="${escapeHtml(option.value)}">${escapeHtml(option.textContent)}</button>`).join('')}</div>`;
    select.after(picker);
    picker.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
      select.value = button.dataset.cycleValue;
      rootElement._loopCycle = select.value;
      rootElement._loopFollow = false;
      picker.querySelector('summary').textContent = `${button.textContent} ⌄`;
      picker.open = false;
      update();
    }));
    select.addEventListener('change', () => { rootElement._loopCycle = select.value; rootElement._loopFollow = false; update(); });
    follow.addEventListener('click', () => {
      rootElement._loopFollow = true;
      rootElement._loopPhase = activePhase || rootElement._loopPhase;
      rootElement._loopCycle = select.value = 'all';
      picker.querySelector('summary').textContent = 'All cycles ⌄';
      update();
    });
    rootElement.querySelectorAll('[data-loop-stage]').forEach(node => {
      const activate = () => { rootElement._loopFollow = false; rootElement._loopPhase = node.dataset.loopStage; update(); };
      node.addEventListener('click', activate);
      node.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(); }
      });
    });
    update();
    return () => hydrate();
  }

  function bindUnifiedDropdowns(rootElement) {
    rootElement._dropdownObserver?.disconnect();
    rootElement._dropdownAbort?.abort();
    const abort = new AbortController();
    rootElement._dropdownAbort = abort;
    const scan = () => {
      rootElement.querySelectorAll('select:not([hidden]):not([multiple])').forEach(select => {
        if (select.classList.contains('protein-native-select') || getComputedStyle(select).display === 'none') return;
        const picker = document.createElement('details');
        picker.className = 'protein-unified-picker';
        const summary = document.createElement('summary');
        summary.setAttribute('aria-label', select.getAttribute('aria-label') || select.title || 'Choose option');
        const menu = document.createElement('div');
        menu.className = 'protein-unified-menu';
        menu.setAttribute('role', 'listbox');
        menu.setAttribute('popover', 'manual');
        picker.append(summary, menu);
        select.after(picker);
        select.classList.add('protein-native-select');
        const sync = () => { summary.textContent = `${select.selectedOptions[0]?.textContent || 'Select'} ⌄`; };
        sync();
        select.addEventListener('change', sync);
        picker.addEventListener('toggle', () => {
          if (!picker.open) { if (menu.matches(':popover-open')) menu.hidePopover(); return; }
          rootElement.querySelectorAll('.protein-unified-picker[open]').forEach(other => { if (other !== picker) other.open = false; });
          menu.replaceChildren();
          [...select.options].forEach(option => {
            const button = document.createElement('button');
            button.type = 'button';
            button.textContent = option.textContent;
            button.disabled = option.disabled;
            button.setAttribute('role', 'option');
            button.setAttribute('aria-selected', String(option.selected));
            button.addEventListener('click', event => {
              event.preventDefault();
              select.value = option.value;
              select.dispatchEvent(new Event('change', {bubbles: true}));
              sync(); picker.open = false; summary.focus();
            });
            menu.appendChild(button);
          });
          const bounds = summary.getBoundingClientRect();
          const spaceBelow = Math.max(0, window.innerHeight - bounds.bottom - 14);
          const spaceAbove = Math.max(0, bounds.top - 14);
          const openBelow = spaceBelow >= Math.min(360, window.innerHeight * 0.55) || spaceBelow >= spaceAbove;
          menu.style.maxHeight = `${Math.min(360, window.innerHeight * 0.55, openBelow ? spaceBelow : spaceAbove)}px`;
          menu.style.width = `${Math.min(Math.max(bounds.width, 170), window.innerWidth - 16)}px`;
          menu.style.left = `${Math.max(8, Math.min(bounds.left, window.innerWidth - parseFloat(menu.style.width) - 8))}px`;
          menu.showPopover();
          const height = menu.getBoundingClientRect().height;
          menu.style.top = `${Math.max(8, openBelow ? bounds.bottom + 6 : bounds.top - height - 6)}px`;
          menu.querySelector('[aria-selected="true"]')?.focus();
        });
        picker.addEventListener('keydown', event => {
          if (event.key === 'Escape') { picker.open = false; summary.focus(); event.preventDefault(); }
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            if (!picker.open) { picker.open = true; return; }
            const options = [...menu.querySelectorAll('button:not(:disabled)')];
            const index = options.indexOf(document.activeElement);
            options[(index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length]?.focus();
          }
        });
      });
    };
    document.addEventListener('pointerdown', event => {
      rootElement.querySelectorAll('.protein-unified-picker[open]').forEach(picker => { if (!picker.contains(event.target)) picker.open = false; });
    }, {signal: abort.signal});
    rootElement.addEventListener('scroll', event => {
      if (event.target instanceof Element && event.target.closest('.protein-unified-menu')) return;
      rootElement.querySelectorAll('.protein-unified-picker[open]').forEach(picker => { picker.open = false; });
    }, {capture: true, signal: abort.signal});
    scan();
    rootElement._dropdownObserver = new MutationObserver(scan);
    rootElement._dropdownObserver.observe(rootElement, {childList: true, subtree: true});
  }

  function mount(rootElement, payload, handlers = {}) {
    const viewState = captureDashboardViewState(rootElement);
    const previousWorkspace = rootElement.querySelector('#proteinCandidateWorkspace');
    const candidateViewState = previousWorkspace
      ? candidatesDashboard.captureDashboardViewState(previousWorkspace) : null;
    const sameRun = rootElement._proteinRunId === payload.run?.taskId;
    if (!sameRun) { rootElement._treePreview?.viewer?.clear(); rootElement._treePreview = null; }
    rootElement._proteinRunId = payload.run?.taskId;
    rootElement.innerHTML = renderProteinDesignDashboard(payload);
    let workspace = rootElement.querySelector('#proteinCandidateWorkspace');
    if (workspace && previousWorkspace && sameRun) {
      workspace.replaceWith(previousWorkspace);
      workspace = previousWorkspace;
    } else if (previousWorkspace) {
      previousWorkspace._proteinViewer?.dispose?.();
    }
    if (workspace) candidatesDashboard.mount(workspace, payload, handlers, sameRun ? candidateViewState : null);
    rootElement.querySelector('#proteinOpenTrace')?.addEventListener('click', () => {
      if (payload.run) handlers.onOpenTrace?.(payload.run);
    });
    const postFilterResults = rootElement.querySelector('.protein-post-filter-results');
    if (postFilterResults) {
      postFilterResults.innerHTML = renderPostFilter(payload.run?.postFilter);
      if (sameRun) postFilterResults.scrollTop = candidateViewState?.postFilterScrollTop || 0;
    }
    bindChartTooltips(rootElement);
    const hydrateOpenActivity = bindDesignLoop(rootElement, payload.run, sameRun);
    if (sameRun) restoreDashboardViewState(rootElement, viewState);
    hydrateOpenActivity();
    bindPostFilterStructures(rootElement);
    bindTree(rootElement, payload.run);
    bindUnifiedDropdowns(rootElement);
  }

  return {
    captureDashboardViewState,
    buildTreeLayout,
    load3Dmol,
    mount,
    orderMetricNames,
    renderProteinDesignDashboard,
    renderActivity,
    renderMarkdown,
    renderRunPicker,
    restoreDashboardViewState,
    selectRunId,
    shortTaskId
  };
});

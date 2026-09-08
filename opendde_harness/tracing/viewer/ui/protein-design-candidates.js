(function initProteinDesignDashboard(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ProteinDesignCandidates = api;
})(typeof window === 'undefined' ? null : window, function buildProteinDesignDashboard() {
  const PROPERTY_DEFINITIONS = [
    { key: 'iptm', label: 'ipTM', paths: ['metrics.iptm'] },
    { key: 'ranking_score', label: 'Ranking score', paths: ['metrics.ranking_score', 'metrics.ranking'] },
    {
      key: 'min_ipa',
      label: 'Min ipAE',
      paths: [
        'metrics.min_ipa',
        'metrics.min_ipae',
        'metrics.min_ipsae',
        'metadata.loss.loss_components.i_pae',
        'metadata.loss.components.i_pae.raw',
        'metrics.ipsae'
      ]
    },
    {
      key: 'loss',
      label: 'Loss',
      paths: [
        'metrics.loss'
      ]
    },
    {
      key: 'contacts',
      label: 'Contacts',
      paths: [
        'metrics.contacts',
        'metrics.cdr_total_contacts',
        'metadata.gate_evidence.cdr_total_contacts',
        'metadata.gate_evidence.total_binder_contacts'
      ]
    },
    {
      key: 'frame_contacts',
      label: 'Frame contacts',
      paths: [
        'metrics.frame_contacts',
        'metrics.framework_total_contacts',
        'metadata.gate_evidence.framework_total_contacts'
      ]
    }
  ];
  const TABLE_SORT_DEFINITIONS = [
    { key: 'cycle', label: 'Cycle' },
    { key: 'sequence', label: 'Sequence' },
    { key: 'candidate_id', label: 'ID' },
    { key: 'target', label: 'Target' },
    { key: 'ranking_score', label: 'Ranking score' },
    { key: 'cdr_contacts', label: 'CDR contacts' },
    { key: 'min_pae', label: 'Min ipAE' }
  ];
  const TABLE_FILTER_DEFINITIONS = [
    { key: 'cycle', label: 'Cycle', integer: true },
    { key: 'iptm', label: 'ipTM' },
    { key: 'ranking_score', label: 'Ranking score' },
    { key: 'min_pae', label: 'Min ipAE' },
    { key: 'loss', label: 'Loss' },
    { key: 'cdr_contacts', label: 'CDR contacts', integer: true },
    { key: 'frame_contacts', label: 'Frame contacts', integer: true }
  ];
  const MOLSTAR_VERSION = '5.11.0';
  const MOLSTAR_BASE_URLS = [
    `https://cdn.jsdelivr.net/npm/molstar@${MOLSTAR_VERSION}/build/viewer`,
    `https://unpkg.com/molstar@${MOLSTAR_VERSION}/build/viewer`
  ];
  const STRUCTURE_COLORS = ['#8f75c9', '#64aeb4', '#df9366', '#70a27b', '#d17694', '#6d89bf'];
  const STRUCTURE_COLOR_MODES = {
    design: 'Design',
    'chain-id': 'Chain',
    'element-symbol': 'Element',
    'plddt-confidence': 'pLDDT',
    'sequence-id': 'Residue order'
  };
  let molstarPromise = null;

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
    return Number(value.toFixed(4)).toString();
  }

  function formatTimestamp(value) {
    const parsed = Date.parse(value || '');
    if (!Number.isFinite(parsed)) return 'Time unavailable';
    const date = new Date(parsed);
    const pad = (part) => String(part).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  function shortTaskId(value) {
    const taskId = String(value || 'unknown');
    return taskId.length <= 12 ? taskId : taskId.slice(0, 12);
  }

  function readPath(value, path) {
    return path.split('.').reduce((current, part) => current?.[part], value);
  }

  function finiteValue(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function consecutiveGroups(positions) {
    const sorted = [...new Set((positions || [])
      .map(Number)
      .filter((value) => Number.isInteger(value) && value >= 0))]
      .sort((left, right) => left - right);
    return sorted.reduce((groups, position) => {
      const current = groups[groups.length - 1];
      if (!current || position !== current[current.length - 1] + 1) groups.push([position]);
      else current.push(position);
      return groups;
    }, []);
  }

  function positionsFromRange(value) {
    if (typeof value === 'number') return Number.isInteger(value) && value >= 0 ? [value] : [];
    if (typeof value !== 'string') return [];
    return value.split(',').flatMap((part) => {
      const match = part.trim().match(/^(\d+)(?::(\d+))?$/);
      if (!match) return [];
      const start = Number(match[1]);
      const end = Number(match[2] ?? match[1]);
      return Array.from({ length: Math.max(0, end - start + 1) }, (_, index) => start + index);
    });
  }

  function normalizeCdrGroups(value) {
    if (typeof value === 'string' || typeof value === 'number') {
      return consecutiveGroups(positionsFromRange(value)).map((positions, index) => ({ label: `CDR${index + 1}`, positions }));
    }
    if (Array.isArray(value)) {
      if (value.some(Array.isArray)) {
        return value.flatMap((positions, index) => {
          const normalized = Array.isArray(positions)
            ? positions.flatMap((position) => positionsFromRange(position))
            : positionsFromRange(positions);
          return normalized.length ? [{ label: `CDR${index + 1}`, positions: [...new Set(normalized)].sort((left, right) => left - right) }] : [];
        });
      }
      return consecutiveGroups(value.flatMap((position) => positionsFromRange(position)))
        .map((positions, index) => ({ label: `CDR${index + 1}`, positions }));
    }
    if (!value || typeof value !== 'object') return [];
    const entries = Object.entries(value).filter(([key]) => /cdr/i.test(key));
    return entries
      .sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true }))
      .flatMap(([key, positions], index) => {
        const normalized = Array.isArray(positions)
          ? positions.flatMap((position) => positionsFromRange(position))
          : positionsFromRange(positions);
        const match = key.match(/cdr\D*(\d+)/i);
        return normalized.length
          ? [{ label: `CDR${match?.[1] || index + 1}`, positions: [...new Set(normalized)].sort((left, right) => left - right) }]
          : [];
      });
  }

  function candidateCdrRegions(candidate, run) {
    const source = run?.cdrRegionGroups
      || candidate?.cdrRegionGroups
      || candidate?.metadata?.cdr_region_groups
      || candidate?.metadata?.cdr_regions
;
    if (!source) return [];
    if (Array.isArray(source) || typeof source !== 'object') return normalizeCdrGroups(source);
    if (Object.keys(source).some((key) => /cdr/i.test(key))) return normalizeCdrGroups(source);
    const preferredChains = [
      candidate?.metadata?.chain_id,
      ...Object.keys(candidate?.metadata?.chains || {}),
      ...Object.keys(source)
    ].filter(Boolean);
    const chainId = preferredChains.find((value) => Object.hasOwn(source, value));
    return chainId ? normalizeCdrGroups(source[chainId]) : [];
  }

  function compactSequence(sequence, leading = 12, trailing = 12) {
    if (sequence.length <= leading + trailing + 3) return sequence;
    return `${sequence.slice(0, leading)}...${sequence.slice(-trailing)}`;
  }

  function sequenceRegionAt(regions, position) {
    return regions.find((region) => region.positions.includes(position));
  }

  function renderFullCandidateSequence(sequence, regions) {
    const chunkSize = 44;
    const lines = [];
    for (let start = 0; start < sequence.length; start += chunkSize) {
      const end = Math.min(sequence.length, start + chunkSize);
      const labels = regions.flatMap((region, regionIndex) => {
        const positions = region.positions.filter((position) => position >= start && position < end);
        if (!positions.length) return [];
        const first = Math.min(...positions) - start;
        const last = Math.max(...positions) - start;
        return [`<span class="protein-full-cdr-label protein-full-cdr-${regionIndex % 3 + 1}" style="grid-column:${first + 1} / span ${last - first + 1}">${escapeHtml(region.label)}</span>`];
      }).join('');
      const residues = sequence.slice(start, end).split('').map((residue, index) => {
        const regionIndex = regions.indexOf(sequenceRegionAt(regions, start + index));
        const className = regionIndex >= 0 ? ` class="is-cdr protein-full-cdr-${regionIndex % 3 + 1}"` : '';
        return `<span${className}>${escapeHtml(residue)}</span>`;
      }).join('');
      const columns = `grid-template-columns:repeat(${end - start},var(--protein-sequence-cell-width))`;
      lines.push(`<div class="protein-full-sequence-line"><div class="protein-full-sequence-labels" style="${columns}">${labels}</div><div class="protein-full-sequence-residues" style="${columns}">${residues}</div></div>`);
    }
    return lines.join('');
  }

  function renderCandidateSequence(candidate, run) {
    const sequence = candidate.sequence || 'Sequence unavailable';
    const regions = candidateCdrRegions(candidate, run);
    if (!regions.length) return `<code aria-label="${escapeHtml(sequence)}">${escapeHtml(compactSequence(sequence, 40, 10))}</code>`;
    const parts = [escapeHtml(sequence.slice(0, 10))];
    regions.forEach((region, index) => {
      const residues = region.positions.map(position => sequence[position] || '').join('');
      parts.push(`<span class="protein-inline-cdr protein-full-cdr-${index % 3 + 1}" title="${escapeHtml(region.label)}"><small>${escapeHtml(region.label)}</small>${escapeHtml(residues.length > 10 ? residues.slice(0, 7) + '…' + residues.slice(-2) : residues)}</span>`);
    });
    parts.push(escapeHtml(sequence.slice(-6)));
    return `<code class="protein-inline-cdr-sequence" aria-label="${escapeHtml(sequence)}">${parts.join('…')}</code>`;
  }

  function renderSequencePreview(candidate, run) {
    const sequence = candidate.sequence || 'Sequence unavailable';
    const regions = candidateCdrRegions(candidate, run)
      .map((region) => ({ ...region, positions: region.positions.filter((position) => position < sequence.length) }))
      .filter((region) => region.positions.length);
    return `<aside class="protein-sequence-preview" role="tooltip" aria-hidden="true"><header><strong>Full sequence</strong><span>${escapeHtml(sequence.length)} residues</span></header><div>${renderFullCandidateSequence(sequence, regions)}</div></aside>`;
  }

  async function copyText(value) {
    try {
      await navigator.clipboard.writeText(value);
      return;
    } catch {}
    const textarea = document.createElement('textarea');
    textarea.value = value;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    textarea.remove();
    if (!copied) throw new Error('Copy is unavailable');
  }

  function propertyValue(candidate, definition) {
    for (const path of definition.paths) {
      const value = finiteValue(readPath(candidate, path));
      if (value !== null) return value;
    }
    return null;
  }

  function propertyDefinition(key) {
    return PROPERTY_DEFINITIONS.find((definition) => definition.key === key);
  }

  function candidateTableValue(candidate, run, key) {
    if (key === 'cycle') return finiteValue(Number(candidate.cycle));
    if (key === 'sequence') return candidate.sequence || '';
    if (key === 'candidate_id') return candidate.candidateId || '';
    if (key === 'target') return run.target || '';
    if (key === 'cdr_contacts') return propertyValue(candidate, propertyDefinition('contacts'));
    if (key === 'min_pae') return propertyValue(candidate, propertyDefinition('min_ipa'));
    const definition = propertyDefinition(key);
    return definition ? propertyValue(candidate, definition) : null;
  }

  function candidateTableAttributes(candidate, run) {
    const values = Object.fromEntries(
      [...new Set([...TABLE_SORT_DEFINITIONS, ...TABLE_FILTER_DEFINITIONS].map((definition) => definition.key))]
        .map((key) => [key, candidateTableValue(candidate, run, key)])
    );
    const attributes = Object.entries(values).map(([key, value]) =>
      `data-table-${key.replaceAll('_', '-')}="${escapeHtml(value ?? '')}"`
    );
    const searchable = [candidate.sequence, candidate.candidateId, run.target].filter(Boolean).join(' ').toLowerCase();
    return `${attributes.join(' ')} data-table-search="${escapeHtml(searchable)}"`;
  }

  function tableFilterDomain(run, definition) {
    const values = allCandidates(run)
      .map((candidate) => candidateTableValue(candidate, run, definition.key))
      .filter((value) => typeof value === 'number' && Number.isFinite(value));
    if (!values.length) return { min: 0, max: 1, step: definition.integer ? 1 : 0.01 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min;
    return {
      min,
      max,
      step: definition.integer ? 1 : Math.max(0.0001, Number((span / 100 || 0.01).toPrecision(2)))
    };
  }

  function renderCandidateFilters(run) {
    const filters = TABLE_FILTER_DEFINITIONS.map((definition) => {
      const domain = tableFilterDomain(run, definition);
      return `<label class="protein-filter-range" data-filter-row="${escapeHtml(definition.key)}"><span><strong>${escapeHtml(definition.label)}</strong><output>${escapeHtml(formatNumber(domain.min))} – ${escapeHtml(formatNumber(domain.max))}</output></span><div><input type="range" min="${domain.min}" max="${domain.max}" step="${domain.step}" value="${domain.min}" data-candidate-filter="${escapeHtml(definition.key)}" data-filter-bound="min" data-domain-min="${domain.min}" data-domain-max="${domain.max}"><input type="range" min="${domain.min}" max="${domain.max}" step="${domain.step}" value="${domain.max}" data-candidate-filter="${escapeHtml(definition.key)}" data-filter-bound="max" data-domain-min="${domain.min}" data-domain-max="${domain.max}"></div><small><span>${escapeHtml(formatNumber(domain.min))}</span><span>${escapeHtml(formatNumber(domain.max))}</span></small></label>`;
    }).join('');
    return `<details class="protein-candidate-filters"><summary><span aria-hidden="true">≡</span> Filters <b hidden>0</b></summary><div class="protein-filter-menu"><header><strong>Filter candidates</strong><button type="button" data-filter-reset>Reset</button></header><input type="search" data-candidate-search placeholder="Search sequence, ID, or target"><p><span data-filter-result-count>${escapeHtml(allCandidates(run).length)}</span> / ${escapeHtml(allCandidates(run).length)} candidates</p>${filters}</div></details>`;
  }

  function renderCandidateControls(run) {
    const options = TABLE_SORT_DEFINITIONS.map((definition) => `<option value="${escapeHtml(definition.key)}">${escapeHtml(definition.label)}</option>`).join('');
    return `<div class="protein-table-controls"><label>View <select data-result-view><option value="design">Design</option><option value="post-filter">Post-filter</option></select></label><label data-design-only>Sort by <select data-candidate-sort>${options}</select></label><button type="button" data-sort-direction value="desc" title="Reverse sort order">DESC ↓</button>${renderCandidateFilters(run)}</div>`;
  }

  function candidateKey(candidate) {
    return `${candidate.cycle}:${candidate.candidateId}`;
  }

  function rankingValue(candidate) {
    return finiteValue(candidate.metrics?.ranking_score) ?? finiteValue(candidate.metrics?.ranking) ?? null;
  }

  function compareCandidates(left, right, run) {
    const leftRanking = rankingValue(left);
    const rightRanking = rankingValue(right);
    if (leftRanking !== null || rightRanking !== null) {
      if (leftRanking === null) return 1;
      if (rightRanking === null) return -1;
      if (rightRanking !== leftRanking) return rightRanking - leftRanking;
    }
    const leftObjective = finiteValue(left.objective);
    const rightObjective = finiteValue(right.objective);
    if (leftObjective === null) return 1;
    if (rightObjective === null) return -1;
    return run.minimize === false ? rightObjective - leftObjective : leftObjective - rightObjective;
  }

  function candidateGroups(run) {
    return (run.cycles || [])
      .map((cycle) => ({
        cycle: cycle.cycle,
        candidates: (cycle.candidates || []).slice().sort((left, right) => compareCandidates(left, right, run))
      }))
      .filter((group) => group.candidates.length)
      .sort((left, right) => right.cycle - left.cycle);
  }

  function allCandidates(run) {
    return candidateGroups(run).flatMap((group) => group.candidates);
  }

  function cycleLeaders(run) {
    return candidateGroups(run).map((group) => group.candidates[0]);
  }

  function chartCandidates(run, selectedKeys) {
    const selected = selectedKeys || new Set();
    const visibleKeys = new Set(cycleLeaders(run).map(candidateKey));
    selected.forEach((key) => visibleKeys.add(key));
    return allCandidates(run).filter((candidate) => visibleKeys.has(candidateKey(candidate)));
  }

  function defaultSelectedKeys(run) {
    return new Set();
  }

  function renderRunPicker(runs, selectedRunId) {
    const selected = runs.find((item) => item.taskId === selectedRunId) || runs[0];
    const options = runs.map((item) => `
      <button type="button" class="protein-run-option${item.taskId === selectedRunId ? ' is-selected' : ''}" data-run-id="${escapeHtml(item.taskId)}" role="option">
        <span><strong>${escapeHtml(item.target || 'Unknown target')}</strong><code title="${escapeHtml(item.taskId)}">${escapeHtml(shortTaskId(item.taskId))}</code></span>
        <small>${escapeHtml(item.status || 'unknown')} · ${escapeHtml(formatTimestamp(item.startTime))}</small>
      </button>`).join('');
    return `<details class="protein-run-picker" id="proteinDesignRunPicker">
      <summary><span class="protein-run-icon" aria-hidden="true">◉</span><strong>${escapeHtml(selected?.target || 'Protein design')}</strong><span>${escapeHtml(selected?.status || 'unknown')}</span><span aria-hidden="true">⌄</span></summary>
      <div class="protein-run-menu" role="listbox">${options}</div>
    </details>`;
  }

  function renderCandidateRow(candidate, selectedKeys, isLeader, run, extraMetrics) {
    const key = candidateKey(candidate);
    const checked = selectedKeys.has(key);
    const ranking = rankingValue(candidate);
    const cdrContacts = candidateTableValue(candidate, run, 'cdr_contacts');
    const minPae = candidateTableValue(candidate, run, 'min_pae');
    const sequence = candidate.sequence || 'Sequence unavailable';
    const missingProperties = PROPERTY_DEFINITIONS.filter((definition) => propertyValue(candidate, definition) === null).map((definition) => definition.label);
    return `<div class="protein-candidate-row${isLeader ? ' is-cycle-leader' : ''}${missingProperties.length ? ' has-missing-properties' : ''}" data-candidate-key="${escapeHtml(key)}" data-candidate-id="${escapeHtml(candidate.candidateId)}" ${candidateTableAttributes(candidate, run)} tabindex="0">
      <label class="protein-candidate-check" title="Compare this sequence">
        <input type="checkbox" data-candidate-select="${escapeHtml(key)}"${checked ? ' checked' : ''}>
        <span aria-hidden="true"></span>
      </label>
      <div class="protein-candidate-cycle"><span>${escapeHtml(candidate.cycle)}</span></div>
      <div class="protein-candidate-id"><code title="${escapeHtml(candidate.candidateId)}">${escapeHtml(candidate.candidateId)}</code></div>
      <div class="protein-candidate-target" title="${escapeHtml(run.target || 'Unknown target')}">${escapeHtml(run.target || '-')}</div>
      <div class="protein-candidate-sequence"><div class="protein-candidate-sequence-main"><span class="protein-sequence-summary" tabindex="0">${renderCandidateSequence(candidate, run)}</span><button type="button" data-copy-sequence="${escapeHtml(sequence)}" title="Copy sequence">Copy</button>${renderSequencePreview(candidate, run)}</div><small>${escapeHtml(sequence.length)} residues${isLeader ? ' · cycle leader' : ''}${missingProperties.length ? ` · <span title="Missing: ${escapeHtml(missingProperties.join(', '))}">metrics incomplete</span>` : ''}</small></div>
      <div class="protein-candidate-score"><strong>${escapeHtml(formatNumber(ranking))}</strong></div>
      <div class="protein-candidate-metric"><strong>${escapeHtml(formatNumber(cdrContacts))}</strong></div>
      <div class="protein-candidate-metric"><strong>${escapeHtml(formatNumber(minPae))}</strong></div>
      ${extraMetrics.map(definition => `<div class="protein-candidate-metric"><strong>${escapeHtml(formatNumber(propertyValue(candidate, definition)))}</strong></div>`).join('')}
    </div>`;
  }

  function renderCandidateTable(run, selectedKeys) {
    const groups = candidateGroups(run);
    const extraMetrics = availableProperties(run).filter(definition => !['ranking_score', 'contacts', 'min_ipa'].includes(definition.key));
    if (!groups.length) return '<div class="protein-empty"><h2>No candidate sequences</h2><p>Candidate rows appear when the first cycle is scored.</p></div>';
    const rows = groups.map((group) => {
      const [leader, ...others] = group.candidates;
      const otherRows = others.map((candidate) => renderCandidateRow(candidate, selectedKeys, false, run, extraMetrics)).join('');
      return `<section class="protein-cycle-group" data-cycle="${escapeHtml(group.cycle)}">
        ${renderCandidateRow(leader, selectedKeys, true, run, extraMetrics)}
        ${others.length ? `<details class="protein-cycle-more"><summary><div class="protein-cycle-more-label">View other sequences <span data-other-count>${escapeHtml(others.length)}</span></div></summary><div>${otherRows}</div></details>` : ''}
      </section>`;
    }).join('');
    return `<div class="protein-candidate-table" style="--candidate-columns:28px 42px 180px 105px 450px repeat(${3 + extraMetrics.length}, 105px);--candidate-table-width:${825 + (3 + extraMetrics.length) * 105}px">
      <div class="protein-candidate-head"><span></span><span>Cycle</span><span>ID</span><span>Target</span><span>Sequence</span><span>Ranking score</span><span>CDR contacts</span><span>Min ipAE</span>${extraMetrics.map(definition => `<span>${escapeHtml(definition.label)}</span>`).join('')}</div>
      <div class="protein-candidate-body">${rows}</div>
    </div>`;
  }

  function propertyDomain(candidates, definition) {
    const values = candidates.map((candidate) => propertyValue(candidate, definition)).filter((value) => value !== null);
    if (!values.length) return { min: 0, max: 1 };
    const observedMax = Math.max(0, ...values);
    if (definition.key === 'contacts' || definition.key === 'frame_contacts') {
      return { min: 0, max: Math.max(5, Math.ceil(observedMax / 5) * 5) };
    }
    return { min: 0, max: Math.max(1, Math.ceil(observedMax)) };
  }

  function propertyTicks(axis, intervals = 5) {
    return Array.from({ length: intervals + 1 }, (_, index) => axis.max * (1 - index / intervals));
  }

  function minIpaColor(value, max) {
    return propertyColor(value, { min: 0, max });
  }

  function observedPropertyDomain(candidates, definition) {
    const values = candidates.map((candidate) => propertyValue(candidate, definition)).filter((value) => value !== null);
    if (!values.length) return { min: null, max: null };
    return { min: Math.min(...values), max: Math.max(...values) };
  }

  function propertyColor(value, domain) {
    if (value === null || domain.min === null || domain.max === null) return '#8a8f98';
    const span = domain.max - domain.min;
    const ratio = span === 0 ? 0.5 : Math.max(0, Math.min(1, (value - domain.min) / span));
    const low = [255, 122, 26];
    const high = [145, 61, 224];
    const channels = low.map((channel, index) => Math.round(channel + (high[index] - channel) * ratio));
    return `rgb(${channels.join(' ')})`;
  }

  function parallelGeometry(candidates, definitions = PROPERTY_DEFINITIONS, width = 920, height = 300) {
    const left = 44;
    const right = 88;
    const top = 50;
    const bottom = 48;
    const step = (width - left - right) / Math.max(1, definitions.length - 1);
    const axes = definitions.map((definition, index) => ({
      ...definition,
      ...propertyDomain(candidates, definition),
      x: definitions.length === 1 ? (left + width - right) / 2 : left + index * step
    }));
    const y = (value, axis) => top + ((axis.max - value) / Math.max(Number.EPSILON, axis.max - axis.min)) * (height - top - bottom);
    return { width, height, top, bottom, axes, y };
  }

  function smoothPath(points) {
    if (!points.length) return '';
    let path = `M ${points[0].x} ${points[0].y}`;
    for (let index = 1; index < points.length; index += 1) {
      const previous = points[index - 1];
      const current = points[index];
      const middle = (previous.x + current.x) / 2;
      path += ` C ${middle} ${previous.y}, ${middle} ${current.y}, ${current.x} ${current.y}`;
    }
    return path;
  }

  function renderParallelCoordinates(run, selectedKeys, enteringKey = null, definitions = PROPERTY_DEFINITIONS, reflow = false) {
    const allRunCandidates = allCandidates(run);
    const candidates = chartCandidates(run, selectedKeys);
    const selected = candidates.filter((candidate) => selectedKeys.has(candidateKey(candidate)));
    const geometry = parallelGeometry(allRunCandidates, definitions);
    const colorDefinition = definitions[definitions.length - 1];
    const colorDomain = observedPropertyDomain(candidates, colorDefinition);
    const axes = geometry.axes.map((axis) => {
      const ticks = propertyTicks(axis).map((value) => {
        const y = geometry.y(value, axis);
        return `<g class="protein-property-tick"><line x1="-4" x2="4" y1="${y}" y2="${y}"/><text x="-8" y="${y + 3}" text-anchor="end">${escapeHtml(formatNumber(value))}</text></g>`;
      }).join('');
      return `<g class="protein-property-axis${reflow ? ' is-reflowing' : ''}" data-axis-key="${escapeHtml(axis.key)}" transform="translate(${axis.x} 0)">
        <text class="protein-property-label" x="0" y="18" text-anchor="middle">${escapeHtml(axis.label)}</text>
        <line class="protein-property-spine" x1="0" x2="0" y1="${geometry.top}" y2="${geometry.height - geometry.bottom}"/>
        ${ticks}
      </g>`;
    }).join('');
    const orderedCandidates = selected.length
      ? candidates.filter((candidate) => !selectedKeys.has(candidateKey(candidate))).concat(selected)
      : candidates;
    const paths = orderedCandidates.map((candidate) => {
      const key = candidateKey(candidate);
      const isSelected = selectedKeys.has(key);
      const missing = [];
      const points = geometry.axes.map((axis) => {
        const value = propertyValue(candidate, axis);
        if (value === null) missing.push(axis.label);
        return { x: axis.x, y: geometry.y(value ?? axis.min, axis) };
      });
      const colorValue = propertyValue(candidate, colorDefinition);
      const color = propertyColor(colorValue, colorDomain);
      const title = `${candidate.candidateId} · cycle ${candidate.cycle} · ${colorDefinition.label} ${formatNumber(colorValue)}${missing.length ? ` · unavailable: ${missing.join(', ')}` : ''}`;
      const stateClass = selected.length ? (isSelected ? ' is-selected' : ' is-muted') : ' is-all';
      const path = smoothPath(points);
      return `<g class="protein-candidate-line-control" data-line-key="${escapeHtml(key)}" role="button" tabindex="0" aria-pressed="${isSelected}" style="--candidate-color:${color}"><path class="protein-candidate-line-hit" d="${path}"/><path class="protein-candidate-line${stateClass}${key === enteringKey ? ' is-entering' : ''}${reflow ? ' is-reflowing' : ''}${missing.length ? ' has-missing-values' : ''}" pathLength="1" d="${path}"><title>${escapeHtml(title)}</title></path></g>`;
    }).join('');
    const scaleHeight = geometry.height - geometry.top - geometry.bottom;
    const candidateCount = candidates.length;
    const summary = selected.length
      ? `<span class="protein-property-summary-item is-muted"><i aria-hidden="true"></i><strong>Candidates (${candidateCount})</strong></span><span class="protein-property-summary-item is-selected"><i aria-hidden="true"></i><strong>Selected candidates (${selected.length})</strong></span>`
      : `<span class="protein-property-summary-item is-selected"><i aria-hidden="true"></i><strong>Candidates (${candidateCount})</strong></span>`;
    return `<div class="protein-properties-chart" data-selected-count="${selected.length}">
      <svg viewBox="0 0 ${geometry.width} ${geometry.height}" role="img" aria-label="Candidate properties">
        <defs><linearGradient id="proteinPropertyColorGradient" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#913de0"/><stop offset="1" stop-color="#ff7a1a"/></linearGradient></defs>
        <g class="protein-property-lines">${paths}</g>
        <g class="protein-property-axes">${axes}</g>
        <g class="protein-property-color-scale" data-color-property="${escapeHtml(colorDefinition.key)}" transform="translate(${geometry.width - 22} ${geometry.top})">
          <text x="12" y="-12" text-anchor="end">${escapeHtml(colorDefinition.label)}</text>
          <rect width="12" height="${scaleHeight}" rx="3" fill="url(#proteinPropertyColorGradient)"/>
        </g>
      </svg>
      <div class="protein-property-summary">${summary}</div>
    </div>`;
  }

  function metricLabel(value) {
    const labels = {iptm: 'ipTM', ptm: 'pTM', plddt: 'pLDDT', iplddt: 'ipLDDT', pae: 'PAE', ipae: 'ipAE', ipsae: 'ipSAE', pde: 'PDE', cdr: 'CDR', cdr1: 'CDR1', cdr2: 'CDR2', cdr3: 'CDR3', rmsd: 'RMSD', esm2: 'ESM2'};
    const text = String(value || '').replace(/^confidence\./, '').replace(/[_-]+/g, ' ');
    return text.split(' ').map((word, index) => labels[word.toLowerCase()] || (index === 0 ? word.charAt(0).toUpperCase() + word.slice(1).toLowerCase() : word.toLowerCase())).join(' ');
  }

  function availableProperties(run) {
    const definitions = [...PROPERTY_DEFINITIONS];
    const known = new Set(definitions.map(definition => definition.paths[0]));
    for (const candidate of run ? allCandidates(run) : []) {
      for (const [key, value] of Object.entries(candidate.metrics || {})) {
        const path = `metrics.${key}`;
        if (typeof value !== 'number' || !Number.isFinite(value) || known.has(path)) continue;
        known.add(path);
        definitions.push({key: `metric:${key}`, label: metricLabel(key), paths: [path]});
      }
    }
    return definitions;
  }

  function propertyDefinitions(keys, run) {
    const definitions = new Map(availableProperties(run).map((definition) => [definition.key, definition]));
    return [...(keys || definitions.keys())].map((key) => definitions.get(key)).filter(Boolean);
  }

  function renderPropertyPicker(visiblePropertyKeys, run) {
    const visible = new Set(visiblePropertyKeys);
    const definitions = availableProperties(run);
    const options = definitions.map((definition) => `<label><input type="checkbox" data-property-key="${escapeHtml(definition.key)}"${visible.has(definition.key) ? ' checked' : ''}><span>${escapeHtml(definition.label)}</span></label>`).join('');
    return `<details class="protein-property-picker"><summary><span class="protein-properties-icon" aria-hidden="true">⌘</span><strong>Properties</strong><span>(${visible.size}/${definitions.length})</span><i aria-hidden="true">⌄</i></summary><div class="protein-property-menu"><p>Visible axes</p>${options}<small>Keep at least two properties.</small></div></details>`;
  }

  function renderOverview(run) {
    const candidates = allCandidates(run);
    const values = (key) => {
      const definition = propertyDefinition(key);
      return candidates.map((candidate) => propertyValue(candidate, definition)).filter((value) => value !== null);
    };
    const rankingScores = values('ranking_score');
    const iptmScores = values('iptm');
    const minPaeScores = values('min_ipa');
    const metrics = [
      ['Candidates', candidates.length],
      ['Best ranking', rankingScores.length ? Math.max(...rankingScores) : null],
      ['Best ipTM', iptmScores.length ? Math.max(...iptmScores) : null],
      ['Min ipAE', minPaeScores.length ? Math.min(...minPaeScores) : null]
    ].map(([label, value]) => `<span><small>${escapeHtml(label)}</small><strong>${escapeHtml(formatNumber(value))}</strong></span>`).join('');
    return `<div class="protein-page-title"><div><span class="protein-target-icon" aria-hidden="true">⌬</span><div><h1>${escapeHtml(run.target || 'Protein design')}</h1><p><code>${escapeHtml(shortTaskId(run.taskId))}</code></p></div></div><div class="protein-overview-metrics">${metrics}</div><div class="protein-run-progress"><span class="protein-status-dot"></span><strong>${escapeHtml(run.status || 'unknown')}</strong><span>Cycle ${escapeHtml(run.cycle)} / ${escapeHtml(run.totalCycles || '?')}</span></div></div>`;
  }

  function renderProteinDesignDashboard(payload, selectedKeys = null, visiblePropertyKeys = null) {
    const runs = payload?.runs || [];
    const run = payload?.run || null;
    if (payload?.error) return `<div class="protein-empty protein-error"><h2>Protein-design data unavailable</h2><p>${escapeHtml(payload.error)}</p></div>`;
    if (!runs.length && !run) return '<div class="protein-empty"><h2>No protein-design runs</h2><p>Start a design task from OpenDDE Harness to populate this workspace.</p></div>';
    if (!run) return '<div class="protein-loading">Loading selected run…</div>';
    const resolvedSelectedKeys = selectedKeys || defaultSelectedKeys(run);
    const resolvedPropertyKeys = visiblePropertyKeys || PROPERTY_DEFINITIONS.map((definition) => definition.key);
    const definitions = propertyDefinitions(resolvedPropertyKeys, run);
    return `<div class="protein-dashboard">
      <header class="protein-dashboard-header">${renderOverview(run)}${renderRunPicker(runs, payload.selectedRunId || run.taskId)}</header>
      <main class="protein-candidate-workspace">
        <section class="protein-candidate-panel">
          <header class="protein-panel-toolbar"><div><strong>Design candidates</strong><span>${escapeHtml(run.candidateCount || allCandidates(run).length)} sequences across ${escapeHtml(candidateGroups(run).length)} cycles</span></div>${renderCandidateControls(run)}</header>
          <div class="protein-design-results">${renderCandidateTable(run, resolvedSelectedKeys)}</div><div class="protein-post-filter-results" hidden><!--post-filter-results--></div>
        </section>
        <section class="protein-inspector-panel">
          <div class="protein-structure-pane">
            <header class="protein-inspector-toolbar"><div class="protein-inspector-identity"><strong id="proteinInspectorCandidate">Structure</strong><small id="proteinInspectorCycle"></small><span id="proteinStructureLegend" class="protein-structure-legend"></span><span class="protein-plddt-legend" title="pLDDT confidence: very low, low, confident, very high"><strong>pLDDT</strong><i></i><i></i><i></i><i></i></span></div><div class="protein-structure-actions"><details class="protein-structure-color-menu"><summary>Color by <strong data-structure-color-label>Design</strong><i aria-hidden="true">⌄</i></summary><div role="menu"><button type="button" data-structure-color="design" class="is-active">Design</button><button type="button" data-structure-color="chain-id">Chain</button><button type="button" data-structure-color="element-symbol">Element</button><button type="button" data-structure-color="plddt-confidence">pLDDT</button><button type="button" data-structure-color="sequence-id">Residue order</button></div></details><button type="button" data-structure-action="sidechains" aria-pressed="false" title="Show side chains">Side chains</button><button type="button" data-structure-action="reset" title="Reset view">Reset</button><button type="button" data-structure-action="expand" title="Expand structure">Expand</button></div></header>
            <div id="proteinStructureViewer" class="has-message" data-message="Loading Mol*…"></div>
          </div>
          <div class="protein-properties-pane">
            <header>${renderPropertyPicker(resolvedPropertyKeys, run)}<small data-property-color-label>Colored by ${escapeHtml(definitions.at(-1).label)}</small></header>
            <div id="proteinPropertiesChart">${renderParallelCoordinates(run, resolvedSelectedKeys, null, definitions)}</div>
          </div>
        </section>
      </main>
    </div>`;
  }

  function loadMolstar() {
    if (typeof window === 'undefined') return Promise.reject(new Error('Mol* requires a browser'));
    if (window.molstar?.Viewer) return Promise.resolve(window.molstar);
    if (molstarPromise) return molstarPromise;
    const loadFrom = (baseUrl) => new Promise((resolve, reject) => {
      const stylesheet = document.querySelector(`link[data-molstar-styles="${baseUrl}"]`) || (() => {
        const stylesheet = document.createElement('link');
        stylesheet.rel = 'stylesheet';
        stylesheet.href = `${baseUrl}/molstar.css`;
        stylesheet.dataset.molstarStyles = baseUrl;
        document.head.appendChild(stylesheet);
        return stylesheet;
      })();
      const script = document.createElement('script');
      script.src = `${baseUrl}/molstar.js`;
      script.async = true;
      const timeout = window.setTimeout(() => {
        script.remove();
        stylesheet.remove();
        reject(new Error('Mol* CDN request timed out'));
      }, 12000);
      script.onload = () => {
        window.clearTimeout(timeout);
        if (window.molstar?.Viewer) resolve(window.molstar);
        else reject(new Error('Mol* loaded without exposing its viewer API'));
      };
      script.onerror = () => {
        window.clearTimeout(timeout);
        script.remove();
        stylesheet.remove();
        reject(new Error('Failed to load Mol* from the configured CDN'));
      };
      document.head.appendChild(script);
    });
    molstarPromise = MOLSTAR_BASE_URLS.reduce(
      (attempt, baseUrl) => attempt.catch(() => loadFrom(baseUrl)),
      Promise.reject(new Error('No Mol* CDN attempted'))
    ).catch((error) => {
      molstarPromise = null;
      throw error;
    });
    return molstarPromise;
  }

  async function readStructureArtifact(artifactPath) {
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const response = await fetch(`/api/artifact?path=${encodeURIComponent(artifactPath)}`, { cache: 'no-store' });
        if (!response.ok) {
          const error = new Error(`Structure request failed (${response.status})`);
          if (response.status < 500) throw error;
          lastError = error;
        } else {
          const artifact = await response.json();
          const structure = artifact.parsed || JSON.parse(artifact.content || '{}');
          if (!structure.text || !structure.format) throw new Error('Structure artifact is incomplete');
          return structure;
        }
      } catch (error) {
        lastError = error;
      }
      if (attempt === 0) await new Promise((resolve) => setTimeout(resolve, 280));
    }
    throw lastError || new Error('Structure request failed');
  }

  function molstarFormat(format) {
    const normalized = String(format || '').toLowerCase().replace(/^\./, '');
    if (normalized === 'cif' || normalized === 'mcif') return 'mmcif';
    return normalized;
  }

  function pdbAlphaCarbons(text, chainIds = []) {
    const allowedChains = new Set(chainIds.map(String));
    return String(text || '').split(/\r?\n/).flatMap((line) => {
      if (!line.startsWith('ATOM  ') || line.slice(12, 16).trim() !== 'CA') return [];
      const chainId = line.slice(21, 22).trim();
      if (allowedChains.size && !allowedChains.has(chainId)) return [];
      const coordinates = [
        Number.parseFloat(line.slice(30, 38)),
        Number.parseFloat(line.slice(38, 46)),
        Number.parseFloat(line.slice(46, 54))
      ];
      return coordinates.every(Number.isFinite) ? [coordinates] : [];
    });
  }

  function alignmentChainIds(structure) {
    const targetChains = Array.isArray(structure.target_chain_ids) ? structure.target_chain_ids : [];
    if (targetChains.length) return targetChains.map(String);
    const binderChains = Array.isArray(structure.binder_chain_ids) ? structure.binder_chain_ids : [];
    return binderChains.map(String);
  }

  function pointCentroid(points) {
    const sum = points.reduce((result, point) => [
      result[0] + point[0],
      result[1] + point[1],
      result[2] + point[2]
    ], [0, 0, 0]);
    return sum.map((value) => value / points.length);
  }

  function dominantEigenvector(matrix) {
    const values = matrix.map((row) => [...row]);
    const vectors = Array.from({ length: 4 }, (_, row) => Array.from({ length: 4 }, (_, column) => Number(row === column)));
    for (let iteration = 0; iteration < 48; iteration += 1) {
      let pivotRow = 0;
      let pivotColumn = 1;
      for (let row = 0; row < 4; row += 1) {
        for (let column = row + 1; column < 4; column += 1) {
          if (Math.abs(values[row][column]) > Math.abs(values[pivotRow][pivotColumn])) {
            pivotRow = row;
            pivotColumn = column;
          }
        }
      }
      if (Math.abs(values[pivotRow][pivotColumn]) < 1e-12) break;
      const angle = 0.5 * Math.atan2(
        2 * values[pivotRow][pivotColumn],
        values[pivotColumn][pivotColumn] - values[pivotRow][pivotRow]
      );
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const diagonalRow = values[pivotRow][pivotRow];
      const diagonalColumn = values[pivotColumn][pivotColumn];
      const pivot = values[pivotRow][pivotColumn];
      for (let index = 0; index < 4; index += 1) {
        if (index === pivotRow || index === pivotColumn) continue;
        const rowValue = values[index][pivotRow];
        const columnValue = values[index][pivotColumn];
        values[index][pivotRow] = values[pivotRow][index] = cosine * rowValue - sine * columnValue;
        values[index][pivotColumn] = values[pivotColumn][index] = sine * rowValue + cosine * columnValue;
      }
      values[pivotRow][pivotRow] = cosine * cosine * diagonalRow - 2 * sine * cosine * pivot + sine * sine * diagonalColumn;
      values[pivotColumn][pivotColumn] = sine * sine * diagonalRow + 2 * sine * cosine * pivot + cosine * cosine * diagonalColumn;
      values[pivotRow][pivotColumn] = values[pivotColumn][pivotRow] = 0;
      for (let row = 0; row < 4; row += 1) {
        const rowValue = vectors[row][pivotRow];
        const columnValue = vectors[row][pivotColumn];
        vectors[row][pivotRow] = cosine * rowValue - sine * columnValue;
        vectors[row][pivotColumn] = sine * rowValue + cosine * columnValue;
      }
    }
    const dominantIndex = values.reduce(
      (best, row, index) => row[index] > values[best][best] ? index : best,
      0
    );
    return vectors.map((row) => row[dominantIndex]);
  }

  function rigidAlignment(referencePoints, movingPoints) {
    const pointCount = Math.min(referencePoints.length, movingPoints.length);
    if (pointCount < 3) return null;
    const reference = referencePoints.slice(0, pointCount);
    const moving = movingPoints.slice(0, pointCount);
    const referenceCenter = pointCentroid(reference);
    const movingCenter = pointCentroid(moving);
    const covariance = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
      const source = moving[pointIndex].map((value, axis) => value - movingCenter[axis]);
      const target = reference[pointIndex].map((value, axis) => value - referenceCenter[axis]);
      for (let row = 0; row < 3; row += 1) {
        for (let column = 0; column < 3; column += 1) covariance[row][column] += source[row] * target[column];
      }
    }
    const [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = covariance;
    const quaternion = dominantEigenvector([
      [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
      [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
      [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
      [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz]
    ]);
    const [w, x, y, z] = quaternion;
    const rotation = [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ];
    const rotatedCenter = rotation.map((row) => row.reduce((sum, value, axis) => sum + value * movingCenter[axis], 0));
    return {
      rotation,
      translation: referenceCenter.map((value, axis) => value - rotatedCenter[axis])
    };
  }

  function transformPoint(point, transform) {
    return transform.rotation.map((row, axis) => row.reduce(
      (sum, value, column) => sum + value * point[column],
      transform.translation[axis]
    ));
  }

  function transformPdb(text, transform) {
    return String(text || '').split(/\r?\n/).map((line) => {
      if (!line.startsWith('ATOM  ') && !line.startsWith('HETATM')) return line;
      const point = [
        Number.parseFloat(line.slice(30, 38)),
        Number.parseFloat(line.slice(38, 46)),
        Number.parseFloat(line.slice(46, 54))
      ];
      if (!point.every(Number.isFinite)) return line;
      const [x, y, z] = transformPoint(point, transform);
      return `${line.slice(0, 30)}${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}${line.slice(54)}`;
    }).join('\n');
  }

  function alignStructureArtifacts(items) {
    if (!Array.isArray(items) || items.length < 2) return items || [];
    const referenceStructure = items[0].structure;
    if (molstarFormat(referenceStructure.format) !== 'pdb') return items;
    const referencePoints = pdbAlphaCarbons(referenceStructure.text, alignmentChainIds(referenceStructure));
    if (referencePoints.length < 3) return items;
    return items.map((item, index) => {
      if (!index || molstarFormat(item.structure.format) !== 'pdb') return item;
      const movingPoints = pdbAlphaCarbons(item.structure.text, alignmentChainIds(item.structure));
      const transform = rigidAlignment(referencePoints, movingPoints);
      if (!transform) return item;
      return { ...item, structure: { ...item.structure, text: transformPdb(item.structure.text, transform) } };
    });
  }

  async function createMolstarViewer(stage) {
    const molstar = await loadMolstar();
    stage.innerHTML = '';
    return molstar.Viewer.create(stage, {
      layoutIsExpanded: false,
      layoutShowControls: true,
      layoutShowRemoteState: false,
      layoutShowSequence: true,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      collapseRightPanel: true,
      viewportShowControls: false,
      viewportShowExpand: false,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      viewportShowTrajectoryControls: false,
      viewportBackgroundColor: document.documentElement.dataset.theme === 'dark' ? '#191c24' : '#ffffff',
      config: [[molstar.lib.plugin.PluginConfig.Viewport.ShowXR, 'never']]
    });
  }

  function structureForCandidate(run, candidate) {
    const projected = (run.structures || []).find((structure) =>
      structure.candidateId === candidate.candidateId && Number(structure.cycle) === Number(candidate.cycle)
    );
    if (projected) return projected;
    if (!candidate.structureArtifactPath) return null;
    const binderChains = Object.keys(candidate.metadata?.chains || {});
    const targetChains = Object.keys(candidate.metadata?.fold?.sequences || {}).filter((chain) => !binderChains.includes(chain));
    return {
      candidateId: candidate.candidateId,
      cycle: candidate.cycle,
      artifactPath: candidate.structureArtifactPath,
      targetChainIds: targetChains,
      binderChainIds: binderChains
    };
  }

  function candidateStructureColor(run, candidate) {
    const index = Math.max(0, allCandidates(run).findIndex((item) => candidateKey(item) === candidateKey(candidate)));
    return STRUCTURE_COLORS[index % STRUCTURE_COLORS.length];
  }

  function colorNumber(value) {
    return Number.parseInt(value.slice(1), 16);
  }

  function usesPlddtColor(structure) {
    const theme = structure.color_theme || structure.colorTheme;
    const source = String(structure.source || structure.database || '').toLowerCase();
    return theme === 'plddt-confidence'
      || source === 'afdb'
      || source === 'alphafold'
      || structure.has_plddt === true
      || structure.hasPlddt === true;
  }

  function chainExpression(chainId) {
    return {
      head: { name: 'structure-query.generator.atom-groups' },
      args: {
        'chain-test': {
          head: { name: 'core.rel.eq' },
          args: [
            { head: { name: 'structure-query.atom-property.macromolecular.auth_asym_id' }, args: {} },
            chainId
          ]
        }
      }
    };
  }

  async function addChainRepresentation(viewer, entry, chainId, key, label, props) {
    const component = await viewer.plugin.builders.structure.tryCreateComponentFromExpression(
      entry.cell,
      chainExpression(chainId),
      key,
      { label }
    );
    if (component) await viewer.plugin.builders.structure.representation.addRepresentation(component, props);
  }

  async function applyComplexStyle(viewer, entry, candidate, structure, color, showTarget) {
    const targetChains = Array.isArray(structure.target_chain_ids) ? structure.target_chain_ids.map(String) : [];
    const binderChains = Array.isArray(structure.binder_chain_ids) ? structure.binder_chain_ids.map(String) : [];
    if (!targetChains.length || !binderChains.length) {
      await viewer.plugin.managers.structure.component.updateRepresentationsTheme(entry.components, {
        color: 'uniform',
        colorParams: { value: colorNumber(color) }
      });
      return;
    }
    viewer.plugin.managers.structure.hierarchy.toggleVisibility(entry.components, 'hide');
    const keyBase = candidateKey(candidate).replace(/[^a-zA-Z0-9]/g, '-');
    if (showTarget) {
      for (const chainId of targetChains) {
        await addChainRepresentation(viewer, entry, chainId, `${keyBase}-target-${chainId}`, `Target ${chainId}`, {
          type: 'molecular-surface',
          color: 'uniform',
          colorParams: { value: 0xdfe2e5 },
          typeParams: { alpha: 0.3 }
        });
      }
    }
    for (const chainId of binderChains) {
      await addChainRepresentation(viewer, entry, chainId, `${keyBase}-binder-${chainId}`, `Binder ${chainId}`, {
        type: 'cartoon',
        color: 'uniform',
        colorParams: { value: colorNumber(color) }
      });
    }
  }

  function syncStructureToolbar(rootElement) {
    const mode = rootElement._proteinStructureColorMode || 'design';
    const label = rootElement.querySelector('[data-structure-color-label]');
    if (label) label.textContent = STRUCTURE_COLOR_MODES[mode];
    rootElement.querySelectorAll('[data-structure-color]').forEach((button) => {
      button.classList.toggle('is-active', button.dataset.structureColor === mode);
    });
    const pane = rootElement.querySelector('.protein-structure-pane');
    pane?.classList.toggle('is-colored-by-plddt', mode === 'plddt-confidence' && rootElement._proteinHasPlddt);
    const sidechains = rootElement.querySelector('[data-structure-action="sidechains"]');
    if (sidechains) {
      sidechains.classList.toggle('is-active', rootElement._proteinShowSidechains === true);
      sidechains.setAttribute('aria-pressed', String(rootElement._proteinShowSidechains === true));
      sidechains.title = rootElement._proteinShowSidechains ? 'Hide side chains' : 'Show side chains';
    }
  }

  async function applyStructureColorMode(rootElement, mode) {
    if (!STRUCTURE_COLOR_MODES[mode]) return;
    rootElement._proteinStructureColorMode = mode;
    syncStructureToolbar(rootElement);
    const viewer = rootElement._proteinViewer;
    if (!viewer) return;
    const entries = viewer.plugin.managers.structure.hierarchy.current.structures;
    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      if (mode !== 'design') {
        await viewer.plugin.managers.structure.component.updateRepresentationsTheme(entry.components, { color: mode });
        continue;
      }
      const candidateColor = colorNumber(rootElement._proteinStructureColors?.[index] || STRUCTURE_COLORS[index % STRUCTURE_COLORS.length]);
      for (const component of entry.components) {
        const target = String(component.cell.obj?.label || '').startsWith('Target ');
        await viewer.plugin.managers.structure.component.updateRepresentationsTheme([component], {
          color: 'uniform',
          colorParams: { value: target ? 0xdfe2e5 : candidateColor }
        });
      }
    }
  }

  function currentSidechainComponents(viewer) {
    return viewer.plugin.managers.structure.hierarchy.current.structures.flatMap((entry) =>
      entry.components.filter((component) => component.cell.obj?.label === 'Side chains')
    );
  }

  async function setSidechainVisibility(rootElement, visible) {
    const viewer = rootElement._proteinViewer;
    rootElement._proteinShowSidechains = visible;
    syncStructureToolbar(rootElement);
    if (!viewer) return;
    let components = currentSidechainComponents(viewer);
    if (visible && !components.length) {
      const query = viewer.plugin.query.structure.registry.list.find((item) => item.label === 'Sidechain with Trace');
      if (!query) return;
      const mode = rootElement._proteinStructureColorMode || 'design';
      const entries = [...viewer.plugin.managers.structure.hierarchy.current.structures];
      for (let index = 0; index < entries.length; index += 1) {
        const component = await viewer.plugin.builders.structure.tryCreateComponentFromSelection(
          entries[index].cell,
          query,
          `dashboard-sidechains-${index}`,
          { label: 'Side chains' }
        );
        if (component) {
          await viewer.plugin.builders.structure.representation.addRepresentation(component, {
            type: 'ball-and-stick',
            color: mode === 'design' ? 'uniform' : mode,
            colorParams: mode === 'design' ? { value: colorNumber(STRUCTURE_COLORS[index % STRUCTURE_COLORS.length]) } : undefined,
            typeParams: { ignoreHydrogens: true, sizeFactor: 0.22 }
          });
        }
      }
      components = currentSidechainComponents(viewer);
    }
    if (components.length) viewer.plugin.managers.structure.hierarchy.toggleVisibility(components, visible ? 'show' : 'hide');
  }

  function updateStructureLegend(rootElement, run, candidates, usesPlddt = false) {
    const legend = rootElement.querySelector('#proteinStructureLegend');
    if (!legend) return;
    legend.innerHTML = candidates.map((candidate) => `<span title="${escapeHtml(candidate.candidateId)}">${escapeHtml(candidate.candidateId)}</span>`).join('');
    rootElement._proteinHasPlddt = usesPlddt;
    syncStructureToolbar(rootElement);
  }

  async function showCandidateStructures(rootElement, run, candidates) {
    const stage = rootElement.querySelector('#proteinStructureViewer');
    const candidateLabel = rootElement.querySelector('#proteinInspectorCandidate');
    const cycleLabel = rootElement.querySelector('#proteinInspectorCycle');
    if (!stage) return;
    rootElement._proteinSequenceObserver?.disconnect();
    const highlightCdr = () => {
      stage.querySelectorAll('.msp-sequence-wrapper').forEach(wrapper => {
        const residues = [...wrapper.querySelectorAll('[data-seqid]')];
        const sequence = residues.map(span => span.textContent.replace(/\u200b/g, '').trim()).join('');
        const candidate = candidates.find(item => item.sequence === sequence);
        const regions = candidate ? candidateCdrRegions(candidate, run) : [];
        residues.forEach(span => {
          const index = regions.findIndex(region => region.positions.includes(Number(span.dataset.seqid)));
          span.classList.remove('protein-molstar-cdr-1', 'protein-molstar-cdr-2', 'protein-molstar-cdr-3');
          if (index >= 0) span.classList.add(`protein-molstar-cdr-${index % 3 + 1}`);
        });
      });
    };
    rootElement._proteinSequenceObserver = new MutationObserver(highlightCdr);
    rootElement._proteinSequenceObserver.observe(stage, {childList:true, subtree:true});
    highlightCdr();
    const requestedCandidates = candidates.filter((candidate) => structureForCandidate(run, candidate));
    const signature = JSON.stringify(requestedCandidates.map((candidate) => [
      candidateKey(candidate), structureForCandidate(run, candidate).artifactPath
    ]));
    if (signature === rootElement._proteinDisplayedSignature && rootElement._proteinViewer) {
      updateStructureLegend(rootElement, run, requestedCandidates);
      return;
    }
    const requestId = (rootElement._proteinStructureRequestId || 0) + 1;
    rootElement._proteinStructureRequestId = requestId;
    if (candidateLabel) candidateLabel.textContent = 'Structure';
    if (cycleLabel) cycleLabel.textContent = requestedCandidates.length === 1 ? `Cycle ${requestedCandidates[0].cycle}` : '';
    updateStructureLegend(rootElement, run, requestedCandidates);
    if (!requestedCandidates.length) {
      if (rootElement._proteinViewer) await rootElement._proteinViewer.plugin.clear();
      rootElement._proteinDisplayedSignature = null;
      stage.dataset.message = 'No captured structure for the current selection.';
      stage.classList.add('has-message');
      return;
    }
    stage.dataset.message = 'Loading structures…';
    stage.classList.remove('protein-error');
    stage.classList.add('has-message');
    try {
      if (!rootElement._proteinViewerPromise) rootElement._proteinViewerPromise = createMolstarViewer(stage);
      const viewer = await rootElement._proteinViewerPromise;
      rootElement._proteinViewer = viewer;
      const outcomes = await Promise.allSettled(requestedCandidates.map(async (candidate) => ({
          candidate,
          structure: {
            target_chain_ids: structureForCandidate(run, candidate).targetChainIds || [],
            binder_chain_ids: structureForCandidate(run, candidate).binderChainIds || [],
            ...await readStructureArtifact(structureForCandidate(run, candidate).artifactPath)
          }
        })));
      if (rootElement._proteinStructureRequestId !== requestId) return;
      const structures = outcomes.filter((outcome) => outcome.status === 'fulfilled').map((outcome) => outcome.value);
      if (!structures.length) {
        const failed = outcomes.find((outcome) => outcome.status === 'rejected');
        throw failed?.reason || new Error('No selected structures could be loaded');
      }
      await viewer.plugin.clear();
      const alignedStructures = alignStructureArtifacts(structures);
      const displayedCandidates = [];
      const displayedColors = [];
      for (let structureIndex = 0; structureIndex < alignedStructures.length; structureIndex += 1) {
        const { candidate, structure } = alignedStructures[structureIndex];
        try {
          const existingCount = viewer.plugin.managers.structure.hierarchy.current.structures.length;
          await viewer.loadStructureFromData(structure.text, molstarFormat(structure.format), { dataLabel: candidate.candidateId });
          const loaded = viewer.plugin.managers.structure.hierarchy.current.structures.slice(existingCount);
          const color = candidateStructureColor(run, candidate);
          for (const entry of loaded) {
            await applyComplexStyle(viewer, entry, candidate, structure, color, displayedCandidates.length === 0);
            displayedColors.push(color);
          }
          displayedCandidates.push(candidate);
        } catch (error) {
          console.warn('Structure rendering failed:', candidate.candidateId, error);
        }
      }
      if (!displayedCandidates.length) {
        throw new Error('The selected structure files could not be rendered');
      }
      rootElement._proteinStructureColors = displayedColors;
      updateStructureLegend(
        rootElement,
        run,
        displayedCandidates,
        alignedStructures.some(({ candidate, structure }) => displayedCandidates.includes(candidate) && usesPlddtColor(structure))
      );
      if (outcomes.some((outcome) => outcome.status === 'rejected')) {
        stage.title = 'Some selected structures could not be loaded';
      } else {
        stage.removeAttribute('title');
      }
      if (rootElement._proteinStructureRequestId !== requestId) return;
      await applyStructureColorMode(rootElement, rootElement._proteinStructureColorMode || 'design');
      if (rootElement._proteinShowSidechains) await setSidechainVisibility(rootElement, true);
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const structureSelect = stage.querySelector('.msp-sequence-select > select:first-of-type');
      const lastCandidateId = displayedCandidates[displayedCandidates.length - 1]?.candidateId;
      const lastOption = [...(structureSelect?.options || [])]
        .find((option) => option.textContent.includes(lastCandidateId));
      if (structureSelect && lastOption) {
        const setValue = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
        setValue?.call(structureSelect, lastOption.value);
        structureSelect.dispatchEvent(new Event('change', { bubbles: true }));
      }
      viewer.plugin.managers.camera.reset();
      viewer.handleResize();
      stage.classList.remove('has-message');
      rootElement._proteinDisplayedSignature = signature;
    } catch (error) {
      if (rootElement._proteinStructureRequestId !== requestId) return;
      if (!rootElement._proteinViewer) rootElement._proteinViewerPromise = null;
      stage.dataset.message = error.message;
      stage.classList.add('has-message', 'protein-error');
    }
  }

  function structureCandidates(rootElement, run, byKey) {
    const selected = allCandidates(run).filter((candidate) => rootElement._proteinSelectedKeys.has(candidateKey(candidate)));
    if (selected.length) {
      const lastSelectedKey = rootElement._proteinLastSelectedKey;
      return selected.filter((candidate) => candidateKey(candidate) !== lastSelectedKey)
        .concat(selected.filter((candidate) => candidateKey(candidate) === lastSelectedKey));
    }
    const active = byKey.get(rootElement._proteinActiveCandidateKey);
    return active ? [active] : [];
  }

  function captureDashboardViewState(rootElement) {
    return {
      openMenus: [...rootElement.querySelectorAll('details[open]:not(.protein-cycle-more)')].map((node) => node.id || node.className),
      menuScroll: rootElement.querySelector('.protein-run-menu')?.scrollTop || 0,
      postFilterScrollTop: rootElement.querySelector('.protein-post-filter-results')?.scrollTop || 0,
      expanded: Boolean(rootElement.querySelector('.protein-structure-pane.is-expanded')),
      scrollTop: rootElement.querySelector('.protein-candidate-table')?.scrollTop || 0,
      scrollLeft: rootElement.querySelector('.protein-candidate-table')?.scrollLeft || 0,
      selectedKeys: [...(rootElement._proteinSelectedKeys || [])],
      lastSelectedKey: rootElement._proteinLastSelectedKey || null,
      visiblePropertyKeys: [...(rootElement._proteinVisiblePropertyKeys || [])],
      activeCandidateKey: rootElement._proteinActiveCandidateKey || null,
      openCycles: [...rootElement.querySelectorAll('.protein-cycle-more[open]')].map((details) => details.closest('.protein-cycle-group')?.dataset.cycle).filter(Boolean)
    };
  }

  function restoreDashboardViewState(rootElement, viewState) {
    if (!viewState) return;
    rootElement.querySelectorAll('details').forEach((node) => {
      if ((viewState.openMenus || []).includes(node.id || node.className)) node.open = true;
    });
    const menu = rootElement.querySelector('.protein-run-menu');
    if (menu) menu.scrollTop = viewState.menuScroll || 0;
    if (viewState.expanded) {
      rootElement.querySelector('.protein-structure-pane')?.classList.add('is-expanded');
      const expand = rootElement.querySelector('[data-structure-action="expand"]');
      if (expand) expand.textContent = 'Close';
    }
    const body = rootElement.querySelector('.protein-candidate-table');
    if (body) { body.scrollTop = viewState.scrollTop || 0; body.scrollLeft = viewState.scrollLeft || 0; }
    const openCycles = new Set(viewState.openCycles || []);
    rootElement.querySelectorAll('.protein-cycle-more').forEach((details) => {
      details.open = openCycles.has(details.closest('.protein-cycle-group')?.dataset.cycle);
    });
  }

  function selectRunId(runs, serverSelectedRunId, currentRunId, pinned) {
    const available = new Set((runs || []).map((run) => run.taskId));
    if (pinned && available.has(currentRunId)) return currentRunId;
    if (available.has(serverSelectedRunId)) return serverSelectedRunId;
    return runs?.[0]?.taskId || null;
  }

  function samplePath(path, count = 72) {
    try {
      const length = path.getTotalLength();
      return Array.from({ length: count }, (_, index) => {
        const point = path.getPointAtLength(length * index / (count - 1));
        return { x: point.x, y: point.y };
      });
    } catch {
      return [];
    }
  }

  function polylinePath(points) {
    return points.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ');
  }

  function capturePropertyLayout(chart) {
    return {
      axes: new Map([...chart.querySelectorAll('[data-axis-key]')].map((axis) => [axis.dataset.axisKey, axis.getBoundingClientRect().left])),
      paths: new Map([...chart.querySelectorAll('.protein-candidate-line')].map((path) => [path.closest('[data-line-key]').dataset.lineKey, samplePath(path)]))
    };
  }

  function animatePropertyLayout(rootElement, chart, previousLayout) {
    const animationId = (rootElement._proteinPropertyAnimationId || 0) + 1;
    rootElement._proteinPropertyAnimationId = animationId;
    chart.querySelectorAll('[data-axis-key]').forEach((axis) => {
      const previousX = previousLayout.axes.get(axis.dataset.axisKey);
      if (previousX === undefined || typeof axis.animate !== 'function') return;
      const delta = previousX - axis.getBoundingClientRect().left;
      axis.animate(
        [{ transform: `translateX(${delta}px)`, opacity: 0.55 }, { transform: 'translateX(0)', opacity: 1 }],
        { duration: 420, easing: 'cubic-bezier(.2,.75,.2,1)' }
      );
    });
    const animations = [];
    chart.querySelectorAll('.protein-candidate-line').forEach((path) => {
      const from = previousLayout.paths.get(path.closest('[data-line-key]').dataset.lineKey);
      const to = samplePath(path);
      if (!from?.length || from.length !== to.length) return;
      animations.push({ path, from, to, target: path.getAttribute('d') });
    });
    if (!animations.length || typeof requestAnimationFrame !== 'function') return;
    const startedAt = performance.now();
    const frame = (now) => {
      if (rootElement._proteinPropertyAnimationId !== animationId) return;
      const progress = Math.min(1, (now - startedAt) / 420);
      const eased = 1 - Math.pow(1 - progress, 3);
      animations.forEach(({ path, from, to, target }) => {
        if (progress === 1) {
          path.setAttribute('d', target);
          path.previousElementSibling?.setAttribute('d', target);
          return;
        }
        const currentPath = polylinePath(from.map((point, index) => ({
          x: point.x + (to[index].x - point.x) * eased,
          y: point.y + (to[index].y - point.y) * eased
        })));
        path.setAttribute('d', currentPath);
        path.previousElementSibling?.setAttribute('d', currentPath);
      });
      if (progress < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  function revealCandidateRow(rootElement, key) {
    const row = [...rootElement.querySelectorAll('[data-candidate-key]')]
      .find(item => item.dataset.candidateKey === key);
    if (!row) return false;
    const view = rootElement.querySelector('[data-result-view]');
    if (view && view.value !== 'design') {
      view.value = 'design';
      view.dispatchEvent(new Event('change', {bubbles: true}));
    }
    if (row.hidden || row.closest('.protein-cycle-group')?.hidden) {
      rootElement.querySelector('[data-filter-reset]')?.click();
    }
    const details = row.closest('.protein-cycle-more');
    if (details) details.open = true;
    row.click();
    row.focus({preventScroll: true});
    const table = row.closest('.protein-candidate-table');
    if (table) {
      const headerHeight = table.querySelector('.protein-candidate-head')?.getBoundingClientRect().height || 0;
      const top = table.scrollTop + row.getBoundingClientRect().top - table.getBoundingClientRect().top - headerHeight - 8;
      table.scrollTo({top: Math.max(0, top), left: 0, behavior: 'smooth'});
    }
    return true;
  }

  function bindPropertyLines(rootElement) {
    rootElement.querySelectorAll('.protein-candidate-line-control').forEach((control) => {
      const reveal = () => revealCandidateRow(rootElement, control.dataset.lineKey);
      control.addEventListener('click', reveal);
      control.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          reveal();
        }
      });
    });
  }

  function animateEnteringPropertyLines(chart) {
    const paths = [...chart.querySelectorAll('.protein-candidate-line.is-entering')];
    return Promise.all(paths.map((path) => new Promise((resolve) => {
      if (!path.isConnected || typeof path.getTotalLength !== 'function') {
        resolve();
        return;
      }
      const length = path.getTotalLength();
      if (!Number.isFinite(length) || length <= 0) {
        resolve();
        return;
      }
      path.style.strokeDasharray = `${length}px`;
      path.style.strokeDashoffset = `${length}px`;
      const animation = path.animate(
        [{ strokeDashoffset: `${length}px` }, { strokeDashoffset: '0px' }],
        { duration: 760, easing: 'cubic-bezier(.2,.75,.2,1)', fill: 'forwards' }
      );
      const finish = () => {
        if (path.isConnected) {
          path.classList.remove('is-entering');
          path.style.strokeDasharray = '';
          path.style.strokeDashoffset = '';
        }
        resolve();
      };
      animation.finished.then(finish, finish);
    })));
  }

  function updateProperties(rootElement, run, enteringKey = null, reflow = false) {
    const chart = rootElement.querySelector('#proteinPropertiesChart');
    const definitions = propertyDefinitions(rootElement._proteinVisiblePropertyKeys, run);
    if (!chart) return Promise.resolve();
    const previousLayout = reflow ? capturePropertyLayout(chart) : null;
    chart.innerHTML = renderParallelCoordinates(run, rootElement._proteinSelectedKeys, enteringKey, definitions, reflow);
    const colorLabel = rootElement.querySelector('[data-property-color-label]');
    if (colorLabel) colorLabel.textContent = `Colored by ${definitions.at(-1).label}`;
    bindPropertyLines(rootElement);
    const enteringAnimation = animateEnteringPropertyLines(chart);
    if (previousLayout) {
      const renderedChart = chart.querySelector('.protein-properties-chart');
      renderedChart?.classList.add(definitions.length < previousLayout.axes.size ? 'is-expanding' : 'is-compressing');
      requestAnimationFrame(() => animatePropertyLayout(rootElement, chart, previousLayout));
    }
    return enteringAnimation;
  }

  function updateCandidateSelection(rootElement) {
    rootElement.querySelectorAll('.protein-candidate-row').forEach((row) => {
      row.classList.toggle('is-compared', rootElement._proteinSelectedKeys.has(row.dataset.candidateKey));
    });
  }

  function tableDatasetName(key) {
    return `table${key.split('_').map((part) => part[0].toUpperCase() + part.slice(1)).join('')}`;
  }

  function rowTableValue(row, key) {
    const value = row.dataset[tableDatasetName(key)];
    if (['sequence', 'candidate_id', 'target'].includes(key)) return value || '';
    return value === '' || value == null ? null : Number(value);
  }

  function compareTableRows(left, right, key, direction) {
    const leftValue = rowTableValue(left, key);
    const rightValue = rowTableValue(right, key);
    if (leftValue == null && rightValue == null) return 0;
    if (leftValue == null) return 1;
    if (rightValue == null) return -1;
    const comparison = typeof leftValue === 'string'
      ? leftValue.localeCompare(rightValue, undefined, { numeric: true })
      : leftValue - rightValue;
    return direction === 'asc' ? comparison : -comparison;
  }

  function applyCandidateTableState(rootElement) {
    const state = rootElement._proteinCandidateTableState;
    const body = rootElement.querySelector('.protein-candidate-body');
    if (!state || !body) return;
    const activeRanges = Object.entries(state.ranges || {}).filter(([, range]) =>
      range.min > range.domainMin || range.max < range.domainMax
    );
    const query = state.query.trim().toLowerCase();
    const matches = (row) => {
      if (query && !String(row.dataset.tableSearch || '').includes(query)) return false;
      return activeRanges.every(([key, range]) => {
        const value = rowTableValue(row, key);
        return value !== null && value >= range.min && value <= range.max;
      });
    };
    let visibleCount = 0;
    const groups = [...body.querySelectorAll('.protein-cycle-group')];
    groups.forEach((group) => {
      const leader = [...group.children].find((child) => child.classList.contains('protein-candidate-row'));
      const details = [...group.children].find((child) => child.classList.contains('protein-cycle-more'));
      const otherContainer = details?.querySelector(':scope > div');
      const otherRows = otherContainer ? [...otherContainer.children] : [];
      otherRows.sort((left, right) => compareTableRows(left, right, state.sortKey, state.direction));
      otherRows.forEach((row) => otherContainer.appendChild(row));
      const leaderVisible = Boolean(leader && matches(leader));
      if (leader) leader.hidden = !leaderVisible;
      const visibleOthers = otherRows.filter((row) => {
        const visible = matches(row);
        row.hidden = !visible;
        return visible;
      });
      if (details) {
        details.hidden = !visibleOthers.length;
        if (query || activeRanges.length) details.open = visibleOthers.length > 0;
        const count = details.querySelector('[data-other-count]');
        if (count) count.textContent = String(query || activeRanges.length ? visibleOthers.length : otherRows.length);
      }
      visibleCount += Number(leaderVisible) + visibleOthers.length;
      group.hidden = !leaderVisible && !visibleOthers.length;
      group._proteinSortRow = leaderVisible ? leader : visibleOthers[0] || leader;
    });
    groups.sort((left, right) => compareTableRows(
      left._proteinSortRow,
      right._proteinSortRow,
      state.sortKey,
      state.direction
    ));
    groups.forEach((group) => body.appendChild(group));
    const activeFilterCount = activeRanges.length + Number(Boolean(query));
    const badge = rootElement.querySelector('.protein-candidate-filters summary b');
    if (badge) {
      badge.hidden = activeFilterCount === 0;
      badge.textContent = String(activeFilterCount);
    }
    const resultCount = rootElement.querySelector('[data-filter-result-count]');
    if (resultCount) resultCount.textContent = String(visibleCount);
  }

  function bindCandidateTableControls(rootElement, run) {
    const state = rootElement._proteinCandidateTableState;
    const sort = rootElement.querySelector('[data-candidate-sort]');
    const direction = rootElement.querySelector('[data-sort-direction]');
    const search = rootElement.querySelector('[data-candidate-search]');
    if (!state || !sort || !direction || !search) return;
    sort.value = state.sortKey;
    direction.value = state.direction;
    direction.textContent = state.direction === 'asc' ? 'ASC ↑' : 'DESC ↓';
    search.value = state.query;
    TABLE_FILTER_DEFINITIONS.forEach((definition) => {
      const domain = tableFilterDomain(run, definition);
      const saved = state.ranges[definition.key];
      const range = {
        min: Math.max(domain.min, Math.min(saved?.min ?? domain.min, domain.max)),
        max: Math.max(domain.min, Math.min(saved?.max ?? domain.max, domain.max)),
        domainMin: domain.min,
        domainMax: domain.max
      };
      if (range.min > range.max) [range.min, range.max] = [range.max, range.min];
      state.ranges[definition.key] = range;
      const inputs = [...rootElement.querySelectorAll(`[data-candidate-filter="${definition.key}"]`)];
      inputs.forEach((input) => { input.value = String(range[input.dataset.filterBound]); });
      const output = rootElement.querySelector(`[data-filter-row="${definition.key}"] output`);
      if (output) output.textContent = `${formatNumber(range.min)} – ${formatNumber(range.max)}`;
    });
    sort.addEventListener('change', () => {
      state.sortKey = sort.value;
      applyCandidateTableState(rootElement);
    });
    direction.addEventListener('click', () => {
      state.direction = state.direction === 'asc' ? 'desc' : 'asc';
      direction.value = state.direction;
      direction.textContent = state.direction === 'asc' ? 'ASC ↑' : 'DESC ↓';
      applyCandidateTableState(rootElement);
    });
    search.addEventListener('input', () => {
      state.query = search.value;
      applyCandidateTableState(rootElement);
    });
    rootElement.querySelectorAll('[data-candidate-filter]').forEach((input) => {
      input.addEventListener('input', () => {
        const key = input.dataset.candidateFilter;
        const range = state.ranges[key];
        const paired = [...rootElement.querySelectorAll(`[data-candidate-filter="${key}"]`)]
          .find((item) => item !== input);
        if (input.dataset.filterBound === 'min' && Number(input.value) > Number(paired.value)) paired.value = input.value;
        if (input.dataset.filterBound === 'max' && Number(input.value) < Number(paired.value)) paired.value = input.value;
        const inputs = [...rootElement.querySelectorAll(`[data-candidate-filter="${key}"]`)];
        range.min = Number(inputs.find((item) => item.dataset.filterBound === 'min').value);
        range.max = Number(inputs.find((item) => item.dataset.filterBound === 'max').value);
        const output = rootElement.querySelector(`[data-filter-row="${key}"] output`);
        if (output) output.textContent = `${formatNumber(range.min)} – ${formatNumber(range.max)}`;
        applyCandidateTableState(rootElement);
      });
    });
    rootElement.querySelector('[data-filter-reset]')?.addEventListener('click', () => {
      state.query = '';
      search.value = '';
      TABLE_FILTER_DEFINITIONS.forEach((definition) => {
        const range = state.ranges[definition.key];
        range.min = range.domainMin;
        range.max = range.domainMax;
        rootElement.querySelectorAll(`[data-candidate-filter="${definition.key}"]`).forEach((input) => {
          input.value = String(range[input.dataset.filterBound]);
        });
        const output = rootElement.querySelector(`[data-filter-row="${definition.key}"] output`);
        if (output) output.textContent = `${formatNumber(range.min)} – ${formatNumber(range.max)}`;
      });
      applyCandidateTableState(rootElement);
    });
    applyCandidateTableState(rootElement);
  }

  function mount(rootElement, payload, handlers = {}, savedViewState = null) {
    const run = payload.run;
    const viewState = savedViewState || captureDashboardViewState(rootElement);
    if (!run) {
      rootElement._proteinViewer?.dispose?.();
      rootElement.innerHTML = renderProteinDesignDashboard(payload);
      rootElement._proteinRunId = null;
      return;
    }
    const sameRun = rootElement._proteinRunId === run.taskId;
    const previousStage = sameRun ? rootElement.querySelector('#proteinStructureViewer') : null;
    rootElement._proteinStructureSelectionId = (rootElement._proteinStructureSelectionId || 0) + 1;
    if (!sameRun) {
      rootElement._proteinViewer?.dispose?.();
      rootElement._proteinViewer = null;
      rootElement._proteinViewerPromise = null;
      rootElement._proteinDisplayedSignature = null;
    }
    const availableKeys = new Set(allCandidates(run).map(candidateKey));
    const restoredSelection = new Set((viewState.selectedKeys || []).filter((key) => availableKeys.has(key)));
    rootElement._proteinSelectedKeys = sameRun ? restoredSelection : defaultSelectedKeys(run);
    rootElement._proteinLastSelectedKey = sameRun && rootElement._proteinSelectedKeys.has(viewState.lastSelectedKey)
      ? viewState.lastSelectedKey
      : [...rootElement._proteinSelectedKeys].at(-1) || null;
    const availablePropertyKeys = new Set(availableProperties(run).map((definition) => definition.key));
    const restoredPropertyKeys = (viewState.visiblePropertyKeys || []).filter((key) => availablePropertyKeys.has(key));
    rootElement._proteinVisiblePropertyKeys = new Set(sameRun && restoredPropertyKeys.length >= 2 ? restoredPropertyKeys : PROPERTY_DEFINITIONS.map(definition => definition.key));
    rootElement._proteinCandidateTableState = sameRun && rootElement._proteinCandidateTableState
      ? rootElement._proteinCandidateTableState
      : { sortKey: 'cycle', direction: 'desc', query: '', ranges: {} };
    if (!STRUCTURE_COLOR_MODES[rootElement._proteinStructureColorMode]) rootElement._proteinStructureColorMode = 'design';
    rootElement._proteinShowSidechains = rootElement._proteinShowSidechains === true;
    if (!sameRun) rootElement._proteinResultView = 'design';
    rootElement._proteinRunId = run.taskId;
    rootElement.innerHTML = renderProteinDesignDashboard(payload, rootElement._proteinSelectedKeys, rootElement._proteinVisiblePropertyKeys);
    if (previousStage && rootElement._proteinViewerPromise) {
      rootElement.querySelector('#proteinStructureViewer')?.replaceWith(previousStage);
    }
    syncStructureToolbar(rootElement);
    const candidates = allCandidates(run);
    const byKey = new Map(candidates.map((candidate) => [candidateKey(candidate), candidate]));
    bindCandidateTableControls(rootElement, run);
    const resultView = rootElement.querySelector('[data-result-view]');
    const showResults = () => {
      const postFilter = resultView.value === 'post-filter';
      rootElement._proteinResultView = resultView.value;
      rootElement.querySelector('.protein-design-results').hidden = postFilter;
      rootElement.querySelector('.protein-post-filter-results').hidden = !postFilter;
      rootElement.querySelectorAll('[data-design-only], [data-sort-direction], .protein-candidate-filters').forEach(node => { node.hidden = postFilter; });
      rootElement.querySelector('.protein-panel-toolbar > div > strong').textContent = postFilter ? 'Post-filter results' : 'Design candidates';
    };
    resultView.value = rootElement._proteinResultView || 'design';
    resultView.addEventListener('change', showResults);
    showResults();

    rootElement.querySelectorAll('[data-run-id]').forEach((button) => {
      button.addEventListener('click', () => handlers.onRunChange?.(button.dataset.runId));
    });
    rootElement.querySelectorAll('[data-candidate-select]').forEach((checkbox) => {
      checkbox.addEventListener('click', (event) => event.stopPropagation());
      checkbox.addEventListener('change', () => {
        const key = checkbox.dataset.candidateSelect;
        if (checkbox.checked) {
          rootElement._proteinSelectedKeys.add(key);
          rootElement._proteinLastSelectedKey = key;
        } else {
          rootElement._proteinSelectedKeys.delete(key);
          if (rootElement._proteinLastSelectedKey === key) {
            rootElement._proteinLastSelectedKey = [...rootElement._proteinSelectedKeys].at(-1) || null;
          }
        }
        updateCandidateSelection(rootElement);
        const structureSelectionId = (rootElement._proteinStructureSelectionId || 0) + 1;
        rootElement._proteinStructureSelectionId = structureSelectionId;
        const enteringAnimation = updateProperties(rootElement, run, checkbox.checked ? key : null);
        if (checkbox.checked) {
          enteringAnimation.then(() => {
            if (rootElement._proteinStructureSelectionId !== structureSelectionId) return;
            showCandidateStructures(rootElement, run, structureCandidates(rootElement, run, byKey));
          });
        } else {
          showCandidateStructures(rootElement, run, structureCandidates(rootElement, run, byKey));
        }
      });
    });
    rootElement.querySelectorAll('[data-property-key]').forEach((checkbox) => {
      checkbox.addEventListener('change', () => {
        const key = checkbox.dataset.propertyKey;
        if (checkbox.checked) rootElement._proteinVisiblePropertyKeys.add(key);
        else if (rootElement._proteinVisiblePropertyKeys.size > 2) rootElement._proteinVisiblePropertyKeys.delete(key);
        else checkbox.checked = true;
        const count = rootElement.querySelector('.protein-property-picker summary > span:last-of-type');
        if (count) count.textContent = `(${rootElement._proteinVisiblePropertyKeys.size}/${availablePropertyKeys.size})`;
        updateProperties(rootElement, run, null, true);
      });
    });
    rootElement.querySelectorAll('[data-copy-sequence]').forEach((button) => {
      button.addEventListener('click', async (event) => {
        event.stopPropagation();
        try {
          await copyText(button.dataset.copySequence);
          button.textContent = 'Copied';
        } catch {
          button.textContent = 'Copy failed';
        }
        window.setTimeout(() => {
          button.textContent = 'Copy';
        }, 1200);
      });
    });
    rootElement.querySelectorAll('.protein-sequence-summary').forEach((summary) => {
      const preview = summary.parentElement.querySelector('.protein-sequence-preview');
      if (!preview) return;
      const showPreview = () => {
        preview.classList.add('is-open');
        preview.setAttribute('aria-hidden', 'false');
        const width = Math.min(620, window.innerWidth - 24);
        const anchor = summary.getBoundingClientRect();
        preview.style.width = `${width}px`;
        preview.style.left = `${Math.max(12, Math.min(anchor.left, window.innerWidth - width - 12))}px`;
        const height = preview.getBoundingClientRect().height;
        const top = anchor.top - height - 10;
        preview.style.top = `${top >= 12 ? top : Math.min(window.innerHeight - height - 12, anchor.bottom + 10)}px`;
      };
      const hidePreview = () => {
        preview.classList.remove('is-open');
        preview.setAttribute('aria-hidden', 'true');
      };
      summary.addEventListener('mouseenter', showPreview);
      summary.addEventListener('mouseleave', hidePreview);
      summary.addEventListener('focus', showPreview);
      summary.addEventListener('blur', hidePreview);
    });
    rootElement.querySelectorAll('.protein-candidate-row').forEach((row) => {
      const activate = () => {
        const candidate = byKey.get(row.dataset.candidateKey);
        if (!candidate) return;
        rootElement._proteinActiveCandidateKey = candidateKey(candidate);
        rootElement.querySelectorAll('.protein-candidate-row').forEach((item) => item.classList.toggle('is-active', item.dataset.candidateKey === rootElement._proteinActiveCandidateKey));
        showCandidateStructures(rootElement, run, structureCandidates(rootElement, run, byKey));
      };
      row.addEventListener('click', (event) => {
        if (event.target.closest('.protein-candidate-check, [data-copy-sequence]')) return;
        activate();
      });
      row.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          activate();
        }
      });
    });
    applyCandidateTableState(rootElement);
    if (sameRun) restoreDashboardViewState(rootElement, viewState);
    updateProperties(rootElement, run);
    updateCandidateSelection(rootElement);
    rootElement.querySelector('[data-structure-action="reset"]')?.addEventListener('click', () => {
      rootElement._proteinViewer?.plugin.managers.camera.reset();
    });
    rootElement.querySelectorAll('[data-structure-color]').forEach((button) => {
      button.addEventListener('click', async () => {
        button.closest('details')?.removeAttribute('open');
        await applyStructureColorMode(rootElement, button.dataset.structureColor);
      });
    });
    rootElement.querySelector('[data-structure-action="sidechains"]')?.addEventListener('click', async () => {
      await setSidechainVisibility(rootElement, !rootElement._proteinShowSidechains);
    });
    rootElement.querySelector('[data-structure-action="expand"]')?.addEventListener('click', (event) => {
      const pane = rootElement.querySelector('.protein-structure-pane');
      const expanded = pane?.classList.toggle('is-expanded') || false;
      event.currentTarget.textContent = expanded ? 'Close' : 'Expand';
      requestAnimationFrame(() => {
        rootElement._proteinViewer?.handleResize();
      });
    });
    const activeCandidate = (sameRun && byKey.get(viewState.activeCandidateKey)) || candidateGroups(run)[0]?.candidates[0];
    if (activeCandidate) {
      rootElement._proteinActiveCandidateKey = candidateKey(activeCandidate);
      rootElement.querySelector(`[data-candidate-key="${CSS.escape(rootElement._proteinActiveCandidateKey)}"]`)?.classList.add('is-active');
      showCandidateStructures(rootElement, run, structureCandidates(rootElement, run, byKey));
    }
  }

  return {
    revealCandidateRow,
    PROPERTY_DEFINITIONS,
    alignStructureArtifacts,
    allCandidates,
    candidateGroups,
    candidateCdrRegions,
    compactSequence,
    chartCandidates,
    captureDashboardViewState,
    defaultSelectedKeys,
    cycleLeaders,
    loadMolstar,
    minIpaColor,
    mount,
    pdbAlphaCarbons,
    parallelGeometry,
    propertyValue,
    propertyTicks,
    propertyDefinitions,
    metricLabel,
    renderParallelCoordinates,
    renderFullCandidateSequence,
    renderProteinDesignDashboard,
    renderRunPicker,
    restoreDashboardViewState,
    selectRunId,
    shortTaskId
  };
});

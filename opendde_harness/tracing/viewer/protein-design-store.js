const ACTIVE_STATUSES = new Set(['queued', 'running']);
const CYCLE_SCHEMA = 'protein_design.cycle.v1';

const DEFAULT_LIMITS = Object.freeze({
  runs: 50,
  cycles: 500,
  candidates: 10000,
  events: 1000
});

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function isGateStatusMetric(name) {
  const parts = String(name || '').split('.');
  const leaf = parts[parts.length - 1];
  return leaf === 'gate_passed' || /_gate_(?:passed|failed|skipped)$/i.test(leaf);
}

function flattenFiniteMetrics(metrics, prefix = '', result = {}) {
  if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) return result;
  for (const [key, value] of Object.entries(metrics)) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (finiteNumber(value) != null) {
      result[name] = value;
    } else if (value && typeof value === 'object' && !Array.isArray(value)) {
      flattenFiniteMetrics(value, name, result);
    }
  }
  return result;
}

function parseTime(value) {
  const parsed = Date.parse(value || '');
  return Number.isFinite(parsed) ? parsed : 0;
}

function dedupeSpans(spans) {
  const latest = new Map();
  for (const span of spans || []) {
    if (span && span.spanId) latest.set(span.spanId, span);
  }
  return [...latest.values()];
}

function attributesOf(span) {
  return span?.attributes && typeof span.attributes === 'object' ? span.attributes : {};
}

function taskIdOf(span) {
  const value = attributesOf(span)['protein_design.task_id'];
  return value == null ? null : String(value);
}

function summarizeRun(span) {
  const attributes = attributesOf(span);
  const status = String(attributes['protein_design.status'] || 'unknown');
  const taskId = taskIdOf(span);
  return {
    taskId,
    target: String(attributes['protein_design.target'] || ''),
    computeUrl: attributes['protein_design.compute_url'] || null,
    computeWorkerId: attributes['protein_design.compute_worker_id'] || null,
    status,
    active: ACTIVE_STATUSES.has(status),
    phase: String(attributes['protein_design.phase'] || ''),
    cycle: Number(attributes['protein_design.cycle'] || 0),
    totalCycles: Number(attributes['protein_design.total_cycles'] || 0),
    objectiveKey: String(attributes['protein_design.objective_key'] || 'objective'),
    minimize: attributes['protein_design.minimize'] !== false,
    bestCandidateId: attributes['protein_design.best_candidate_id'] || null,
    bestObjective: finiteNumber(attributes['protein_design.best_objective']),
    error: attributes['protein_design.error'] || null,
    startTime: span.startTime || null,
    endTime: span.endTime || null,
    traceId: span.traceId || null,
    sessionId: attributes['session.id'] || null,
    spanId: span.spanId,
    finalSelectionArtifactPath:
      attributes['protein_design.final_selection.artifact_path'] || null
  };
}

function compareRuns(left, right) {
  if (left.active !== right.active) return left.active ? -1 : 1;
  const timeDifference = parseTime(right.startTime) - parseTime(left.startTime);
  if (timeDifference !== 0) return timeDifference;
  return String(left.taskId).localeCompare(String(right.taskId));
}

function parseArtifact(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value;
  if (typeof value !== 'string') return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function readCycleArtifact(span, readArtifact) {
  const artifactPath = attributesOf(span)['protein_design.cycle.artifact_path'];
  if (!artifactPath) return null;
  try {
    const artifact = parseArtifact(readArtifact(artifactPath));
    return artifact?.schema_version === CYCLE_SCHEMA ? artifact : null;
  } catch {
    return null;
  }
}

function readProgressArtifact(span, readArtifact) {
  const artifactPath = attributesOf(span)['protein_design.progress.artifact_path'];
  if (!artifactPath) return null;
  try {
    const artifact = parseArtifact(readArtifact(artifactPath));
    return artifact && artifact.task_id ? artifact : null;
  } catch {
    return null;
  }
}

function activityEventKey(event) {
  return [
    event?.cycle,
    event?.event_type,
    event?.phase,
    event?.actor,
    event?.tool
  ].map((value) => String(value ?? '')).join('\u0000');
}

function hasActivityDetails(event) {
  return [event?.input_payload, event?.output_payload, event?.metadata, event?.error]
    .some((value) => value != null && value !== '' && (
      typeof value !== 'object' || Object.keys(value).length > 0
    ));
}

function projectActivityEvents(events, limit) {
  const finished = new Set(
    events
      .filter((event) => event.status === 'completed' || event.status === 'failed')
      .map(activityEventKey)
  );
  return events
    .filter((event) => {
      if (event.status === 'started' && finished.has(activityEventKey(event))) return false;
      if (hasActivityDetails(event) || event.status === 'failed') return true;
      if (['fold', 'initial_fold', 'parent_selection'].includes(event.phase)) return true;
      return event.status === 'started' && ['agent', 'skill', 'tool'].includes(event.event_type);
    })
    .slice(-limit);
}

function readFinalSelectionArtifact(summary, readArtifact) {
  const artifactPath = summary?.finalSelectionArtifactPath;
  if (!artifactPath) return null;
  try {
    return parseArtifact(readArtifact(artifactPath));
  } catch {
    return null;
  }
}

function candidateMetrics(candidate, objectiveKey) {
  const metrics = flattenFiniteMetrics(candidate?.metrics);
  const objective = finiteNumber(candidate?.objective);
  if (objective != null) metrics[objectiveKey] = objective;
  return metrics;
}

function candidateRecord(candidate, cycle, objectiveKey) {
  if (!candidate || candidate.candidate_id == null) return null;
  return {
    candidateId: String(candidate.candidate_id),
    cycle,
    sequence: typeof candidate.sequence === 'string' ? candidate.sequence : '',
    objective: finiteNumber(candidate.objective),
    metrics: candidateMetrics(candidate, objectiveKey),
    parentId: candidate.parent_id == null ? null : String(candidate.parent_id),
    skillId: candidate.skill_id == null ? null : String(candidate.skill_id),
    status: candidate.status == null ? 'unknown' : String(candidate.status),
    structureArtifactPath:
      typeof candidate.structure_artifact_path === 'string'
        ? candidate.structure_artifact_path
        : null,
    metadata:
      candidate.metadata && typeof candidate.metadata === 'object' && !Array.isArray(candidate.metadata)
        ? candidate.metadata
        : {}
  };
}

function structureChainRoles(candidate) {
  const metadata = candidate?.metadata || {};
  const binderChains = metadata.chains;
  const foldedChains = metadata.fold?.sequences;
  const binderChainIds = binderChains && typeof binderChains === 'object'
    ? Object.keys(binderChains)
    : [];
  const targetChainIds = foldedChains && typeof foldedChains === 'object'
    ? Object.keys(foldedChains).filter((chainId) => !binderChainIds.includes(chainId))
    : [];
  return { targetChainIds, binderChainIds };
}

function candidateGateDetails(candidate) {
  const metadata = candidate?.metadata || {};
  const evidence = metadata.gate_evidence && typeof metadata.gate_evidence === 'object'
    ? metadata.gate_evidence
    : {};
  const rawPassed = metadata.gate_passed ?? candidate?.metrics?.gate_passed;
  const gatePassed = typeof rawPassed === 'boolean'
    ? rawPassed
    : finiteNumber(rawPassed) == null ? null : Number(rawPassed) > 0;
  const gateMetrics = Object.fromEntries(
    Object.entries(candidate?.metrics || {}).filter(([name, value]) =>
      finiteNumber(value) != null && /cdr|contact|gate|i_con|epitope|framework|hotspot|rmsd/i.test(name)
    )
  );
  for (const [name, value] of Object.entries(evidence)) {
    if (finiteNumber(value) != null && !Object.hasOwn(gateMetrics, name)) {
      gateMetrics[name] = value;
    }
  }
  return { gatePassed, gateMetrics, gateEvidence: evidence };
}

function projectPostFilter(summary, readArtifact) {
  const artifact = readFinalSelectionArtifact(summary, readArtifact);
  if (!artifact) return null;
  const legacySummary = String(artifact.strategy_summary || '');
  const enabled = artifact.post_filter_enabled === true
    || artifact.mode === 'agent'
    || Boolean(artifact.post_filter_error)
    || (/postfilter agent|post-filter/i.test(legacySummary) && !/post-filter disabled/i.test(legacySummary));
  if (!enabled) return null;
  const decisions = Array.isArray(artifact.decisions) ? artifact.decisions : [];
  const structureArtifacts = artifact.structure_artifacts && typeof artifact.structure_artifacts === 'object'
    ? artifact.structure_artifacts
    : {};
  const targetChainIds = Array.isArray(artifact.target_chain_ids)
    ? artifact.target_chain_ids.map(String)
    : [];
  const binderChainIds = Array.isArray(artifact.binder_chain_ids)
    ? artifact.binder_chain_ids.map(String)
    : [];
  const selectedCandidateIds = Array.isArray(artifact.selected_candidate_ids)
    ? artifact.selected_candidate_ids.map(String)
    : [];
  const candidatesById = new Map(
    (Array.isArray(artifact.candidates) ? artifact.candidates : [])
      .filter((candidate) => candidate && candidate.candidate_id)
      .map((candidate) => [String(candidate.candidate_id), candidate])
  );
  return {
    enabled: true,
    executed: artifact.post_filter_executed === true,
    mode: String(artifact.mode || 'unknown'),
    status: artifact.mode === 'failed' || artifact.post_filter_error
      ? 'failed'
      : artifact.post_filter_executed ? 'completed' : 'skipped',
    strategySummary: String(artifact.strategy_summary || ''),
    topK: Number(artifact.post_filter_top_k || selectedCandidateIds.length || 0),
    refoldEnabled: enabled,
    refoldedCandidateCount: Number(artifact.refolded_candidate_count || 0),
    eligibleCandidateCount: Number(artifact.eligible_candidate_count || 0),
    selectedCandidateIds,
    postRefoldError: artifact.post_refold_error || null,
    postFilterError: artifact.post_filter_error || null,
    decisions: decisions.map((decision) => {
      const candidateId = String(decision?.candidate_id || '');
      const candidate = candidatesById.get(candidateId) || {};
      const metrics = Object.fromEntries(
        Object.entries(flattenFiniteMetrics(candidate.metrics || {}))
          .filter(([name]) => !isGateStatusMetric(name))
      );
      return {
        candidateId,
        rank: Number(decision?.rank || 0),
        objective: finiteNumber(decision?.objective),
        sequence: typeof candidate.sequence === 'string' ? candidate.sequence : '',
        metrics,
        passFilter: decision?.pass_filter === true,
        hardEligible: decision?.hard_eligible === true,
        rationale: String(decision?.rationale || ''),
        strengths: Array.isArray(decision?.strengths) ? decision.strengths.map(String) : [],
        risks: Array.isArray(decision?.risks) ? decision.risks.map(String) : [],
        structureArtifactPath: structureArtifacts[candidateId] || null,
        targetChainIds,
        binderChainIds
      };
    })
  };
}

function pointFor(candidate, metric, cycle) {
  const value = candidate?.metrics?.[metric];
  return finiteNumber(value) == null
    ? null
    : { cycle, candidateId: candidate.candidateId, value };
}

function bestScoredCandidate(candidates, minimize) {
  return (candidates || []).reduce((best, candidate) => {
    if (finiteNumber(candidate?.objective) == null) return best;
    if (!best) return candidate;
    return minimize
      ? (candidate.objective < best.objective ? candidate : best)
      : (candidate.objective > best.objective ? candidate : best);
  }, null);
}

function objectiveOnlyCandidate(candidateId, objectiveKey, objective) {
  if (!candidateId || finiteNumber(objective) == null) return null;
  return {
    candidateId: String(candidateId),
    objective,
    metrics: { [objectiveKey]: objective }
  };
}

function projectSelectedRun(summary, cycleSpans, progressSpans, readArtifact, limits) {
  const sortedCycleSpans = cycleSpans
    .slice()
    .sort((left, right) => {
      const a = Number(attributesOf(left)['protein_design.cycle_index'] || 0);
      const b = Number(attributesOf(right)['protein_design.cycle_index'] || 0);
      return a - b || parseTime(left.startTime) - parseTime(right.startTime);
    })
    .slice(0, limits.cycles);
  const cycles = [];
  const candidateById = new Map();
  const metricNames = new Set();
  const structuresByPath = new Map();
  const structureCaptureFailures = [];
  const treeNodesById = new Map();
  const treeEdges = [];
  const cycleSelections = [];
  let populationCandidateIds = new Set();
  let populationSnapshotAvailable = false;
  let candidateCount = 0;
  let failureCount = 0;

  for (const cycleSpan of sortedCycleSpans) {
    const artifact = readCycleArtifact(cycleSpan, readArtifact);
    if (!artifact) continue;
    const cycle = Number(artifact.cycle ?? attributesOf(cycleSpan)['protein_design.cycle_index'] ?? 0);
    const cycleRootId = `cycle:${cycle}`;
    const candidates = [];
    const admittedCandidateIds = new Set(
      Array.isArray(artifact.admitted_candidate_ids)
        ? artifact.admitted_candidate_ids.map(String)
        : []
    );
    const populationActions = artifact.population_actions && typeof artifact.population_actions === 'object'
      ? artifact.population_actions
      : {};
    if (Array.isArray(artifact.population_candidate_ids)) {
      populationSnapshotAvailable = true;
      populationCandidateIds = new Set(artifact.population_candidate_ids.map(String));
    }
    for (const rawCandidate of Array.isArray(artifact.candidates) ? artifact.candidates : []) {
      if (candidateCount >= limits.candidates) break;
      const candidate = candidateRecord(rawCandidate, cycle, summary.objectiveKey);
      if (!candidate) continue;
      candidate.admitted = admittedCandidateIds.has(candidate.candidateId);
      candidate.populationAction = populationActions[candidate.candidateId] == null
        ? null
        : String(populationActions[candidate.candidateId]);
      candidateCount += 1;
      if (/fail|error/i.test(candidate.status)) failureCount += 1;
      candidates.push(candidate);
      candidateById.set(candidate.candidateId, candidate);
      Object.keys(candidate.metrics)
        .filter((name) => !isGateStatusMetric(name))
        .forEach((name) => metricNames.add(name));
      if (candidate.structureArtifactPath && !structuresByPath.has(candidate.structureArtifactPath)) {
        const chainRoles = structureChainRoles(candidate);
        structuresByPath.set(candidate.structureArtifactPath, {
          candidateId: candidate.candidateId,
          cycle,
          artifactPath: candidate.structureArtifactPath,
          objective: candidate.objective,
          sequence: candidate.sequence,
          parentId: candidate.parentId,
          skillId: candidate.skillId,
          populationAction: candidate.populationAction,
          admitted: candidate.admitted,
          ...candidateGateDetails(candidate),
          ...chainRoles
        });
      } else if (candidate.metadata?.structure_artifact_error) {
        structureCaptureFailures.push({
          candidateId: candidate.candidateId,
          cycle,
          error: String(candidate.metadata.structure_artifact_error)
        });
      }
      let parentId = cycleRootId;
      if (candidate.parentId) {
        parentId = `candidate:${candidate.parentId}`;
        if (!treeNodesById.has(parentId)) {
          treeNodesById.set(parentId, {
            id: parentId,
            kind: 'initial',
            candidateId: candidate.parentId,
            cycle: -1,
            label: candidate.parentId
          });
        }
      } else if (!treeNodesById.has(cycleRootId)) {
        treeNodesById.set(cycleRootId, {
          id: cycleRootId,
          kind: 'cycle',
          cycle,
          label: `Cycle ${cycle}`
        });
      }
      const nodeId = `candidate:${candidate.candidateId}`;
      treeNodesById.set(nodeId, {
        id: nodeId,
        kind: 'candidate',
        parentId,
        candidateId: candidate.candidateId,
        cycle,
        objective: candidate.objective,
        skillId: candidate.skillId,
        status: candidate.status
      });
      treeEdges.push({ source: parentId, target: nodeId });
    }
    const declaredCycleBest = candidateById.get(artifact.cycle_best_candidate_id);
    const cycleBest = declaredCycleBest || bestScoredCandidate(candidates, summary.minimize);
    const globalBestId = artifact.global_best_candidate_id || summary.bestCandidateId;
    const globalBestObjective = finiteNumber(
      attributesOf(cycleSpan)['protein_design.global_best_objective']
    );
    const globalBest = candidateById.get(globalBestId) || objectiveOnlyCandidate(
      globalBestId,
      summary.objectiveKey,
      globalBestObjective ?? summary.bestObjective
    );
    cycleSelections.push({ cycleBest, globalBest });
    cycles.push({
      cycle,
      candidateCount: candidates.length,
      cycleBestCandidateId: artifact.cycle_best_candidate_id || null,
      globalBestCandidateId: artifact.global_best_candidate_id || null,
      candidates
    });
  }

  const names = [...metricNames].sort((a, b) => a.localeCompare(b));
  const series = Object.fromEntries(
    names.map((name) => [name, { cycleBest: [], globalBest: [] }])
  );
  for (let index = 0; index < cycles.length; index += 1) {
    const cycle = cycles[index];
    const { cycleBest, globalBest } = cycleSelections[index];
    for (const metric of names) {
      const cyclePoint = pointFor(cycleBest, metric, cycle.cycle);
      const globalPoint = pointFor(globalBest, metric, cycle.cycle);
      if (cyclePoint) series[metric].cycleBest.push(cyclePoint);
      if (globalPoint) series[metric].globalBest.push(globalPoint);
    }
  }

  const allEvents = progressSpans
    .map((span) => readProgressArtifact(span, readArtifact))
    .filter(Boolean)
    .sort((left, right) => parseTime(left.timestamp) - parseTime(right.timestamp));
  const events = projectActivityEvents(allEvents, limits.events);
  const eventCount = events.length;

  const structures = [...structuresByPath.values()].map((structure) => ({
    ...structure,
    inPopulation: populationSnapshotAvailable
      ? populationCandidateIds.has(structure.candidateId)
      : null
  }));
  const structuresByCandidate = new Map(
    structures.map((structure) => [structure.candidateId, structure])
  );
  const postFilter = projectPostFilter(summary, readArtifact);
  if (postFilter) {
    postFilter.decisions = postFilter.decisions.map((decision) => {
      if (decision.structureArtifactPath) return decision;
      const structure = structuresByCandidate.get(decision.candidateId);
      return structure
        ? {
            ...decision,
            structureArtifactPath: structure.artifactPath,
            targetChainIds: structure.targetChainIds,
            binderChainIds: structure.binderChainIds
          }
        : decision;
    });
  }
  const population = [...populationCandidateIds].map((candidateId) => {
    const candidate = candidateById.get(candidateId);
    return candidate
      ? {
          candidateId,
          cycle: candidate.cycle,
          objective: candidate.objective,
          skillId: candidate.skillId,
          structureArtifactPath: candidate.structureArtifactPath,
          ...candidateGateDetails(candidate)
        }
      : { candidateId };
  });

  return {
    ...summary,
    candidateCount,
    events,
    eventCount,
    progressEventCount: allEvents.length,
    failureCount,
    cycles,
    metricNames: names,
    series,
    structures,
    population,
    populationCandidateIds: [...populationCandidateIds],
    populationSnapshotAvailable,
    postFilter,
    structureCaptureFailures,
    tree: { nodes: [...treeNodesById.values()], edges: treeEdges }
  };
}

function projectProteinDesignRuns({
  spans,
  readArtifact,
  selectedRunId = null,
  registeredTaskIds = null,
  limits = {}
}) {
  const resolvedLimits = { ...DEFAULT_LIMITS, ...limits };
  const deduped = dedupeSpans(spans);
  const allowedTaskIds = registeredTaskIds == null
    ? null
    : new Set(registeredTaskIds);
  const runs = deduped
    .filter((span) => {
      const taskId = taskIdOf(span);
      return (
        span.name === 'protein_design.run' &&
        taskId &&
        (allowedTaskIds === null || allowedTaskIds.has(taskId))
      );
    })
    .map(summarizeRun)
    .sort(compareRuns)
    .slice(0, resolvedLimits.runs);
  const selected = runs.find((run) => run.taskId === selectedRunId) || runs[0] || null;
  if (!selected) return { runs: [], selectedRunId: null, run: null };
  const cycleSpans = deduped.filter(
    (span) => span.name === 'protein_design.cycle' && taskIdOf(span) === selected.taskId
  );
  const progressSpans = deduped.filter(
    (span) => span.name === 'protein_design.progress' && taskIdOf(span) === selected.taskId
  );
  return {
    runs,
    selectedRunId: selected.taskId,
    run: projectSelectedRun(selected, cycleSpans, progressSpans, readArtifact, resolvedLimits)
  };
}

module.exports = {
  flattenFiniteMetrics,
  projectProteinDesignRuns
};

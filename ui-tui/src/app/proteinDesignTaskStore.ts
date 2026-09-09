// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { atom } from 'nanostores'
import { readdir, readFile, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

export type ProteinDesignTaskStatus = 'queued' | 'running' | 'completed' | 'failed' | 'stopped'

export interface ProteinDesignProgressPayload {
  task_id: string
  status: 'started' | 'progress' | 'completed' | 'failed'
  cycle?: number | null
  total_cycles?: number | null
  phase?: string
  actor?: string
  skill?: string | null
  tool?: string | null
  summary?: string
  duration_ms?: number | null
  candidate_count?: number | null
}

export interface ProteinDesignTask {
  bestObjective?: number
  computeUrl?: string
  computeWorkerId?: string
  cycle: number
  error?: string
  phase: string
  selectedSkill?: string
  status: ProteinDesignTaskStatus
  target: string
  taskId: string
  totalCycles: number
  updatedAt: number
}

interface ProteinDesignTaskState {
  focusedTaskId: string | null
  tasks: Record<string, ProteinDesignTask>
}

const initialState = (): ProteinDesignTaskState => ({ focusedTaskId: null, tasks: {} })

export const $proteinDesignTasks = atom<ProteinDesignTaskState>(initialState())

const taskRoot = () =>
  process.env.OPENDDE_HARNESS_PROTEIN_DESIGN_ROOT || join(homedir(), '.opendde_harness', 'protein_design')

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

const text = (value: unknown, fallback = '') => (typeof value === 'string' ? value : fallback)
const number = (value: unknown, fallback = 0) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback

const normalizeStatus = (value: unknown): ProteinDesignTaskStatus => {
  const status = text(value).toLowerCase()

  if (
    status === 'queued' ||
    status === 'running' ||
    status === 'completed' ||
    status === 'failed' ||
    status === 'stopped'
  ) {
    return status
  }

  return 'running'
}

const safeTaskId = (value: string) => Boolean(value) && !value.includes('/') && !value.includes('\\') && value !== '..'

const snapshotTask = (value: unknown, updatedAt: number): ProteinDesignTask | null => {
  if (!isRecord(value)) {
    return null
  }

  const taskId = text(value.task_id)

  if (!safeTaskId(taskId)) {
    return null
  }

  const best = isRecord(value.best_candidate) ? value.best_candidate : null

  return {
    ...(best && typeof best.objective === 'number' && Number.isFinite(best.objective)
      ? { bestObjective: best.objective }
      : {}),
    ...(text(value.compute_url) ? { computeUrl: text(value.compute_url) } : {}),
    ...(text(value.compute_worker_id) ? { computeWorkerId: text(value.compute_worker_id) } : {}),
    cycle: number(value.cycle),
    ...(text(value.error) ? { error: text(value.error) } : {}),
    phase: text(value.phase, 'setup'),
    ...(text(value.selected_skill) ? { selectedSkill: text(value.selected_skill) } : {}),
    status: normalizeStatus(value.status),
    target: text(value.target, 'protein design'),
    taskId,
    totalCycles: number(value.total_cycles),
    updatedAt
  }
}

export const applyProteinDesignProgress = (payload: ProteinDesignProgressPayload): boolean => {
  const previous = $proteinDesignTasks.get().tasks[payload.task_id]
  const status: ProteinDesignTaskStatus =
    payload.status === 'completed' ? 'completed' : payload.status === 'failed' ? 'failed' : 'running'
  const task: ProteinDesignTask = {
    ...previous,
    cycle: payload.cycle ?? previous?.cycle ?? 0,
    phase: payload.phase || previous?.phase || 'setup',
    ...(payload.skill ? { selectedSkill: payload.skill } : {}),
    status,
    target: previous?.target || 'protein design',
    taskId: payload.task_id,
    totalCycles: payload.total_cycles ?? previous?.totalCycles ?? 0,
    updatedAt: Date.now()
  }

  $proteinDesignTasks.set({
    ...$proteinDesignTasks.get(),
    tasks: { ...$proteinDesignTasks.get().tasks, [task.taskId]: task }
  })

  return Boolean((!previous || previous.status !== status) && (status === 'completed' || status === 'failed'))
}

export const refreshProteinDesignTasks = async (root = taskRoot()): Promise<ProteinDesignTask[]> => {
  let entries

  try {
    entries = await readdir(root, { withFileTypes: true })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return []
    }
    throw error
  }

  const loaded = (
    await Promise.all(
      entries
        .filter(entry => entry.isDirectory())
        .map(async entry => {
          const path = join(root, entry.name, 'snapshot.json')

          try {
            const [body, info] = await Promise.all([readFile(path, 'utf8'), stat(path)])
            return snapshotTask(JSON.parse(body) as unknown, info.mtimeMs)
          } catch {
            // A worker uses atomic replace, but tolerate malformed/manual task
            // directories so one bad run never hides every healthy task.
            return null
          }
        })
    )
  ).filter((task): task is ProteinDesignTask => task !== null)

  if (loaded.length) {
    const state = $proteinDesignTasks.get()
    const tasks = { ...state.tasks }

    for (const task of loaded) {
      tasks[task.taskId] = task
    }

    $proteinDesignTasks.set({ ...state, tasks })
  }

  return loaded
}

export const startProteinDesignTaskPolling = (intervalMs = 2_000): (() => void) => {
  let stopped = false
  let inFlight = false

  const refresh = async () => {
    if (stopped || inFlight) {
      return
    }
    inFlight = true
    try {
      await refreshProteinDesignTasks()
    } catch {
      // The monitor is informational. A temporarily unavailable task root
      // must never disturb typing or crash the TUI.
    } finally {
      inFlight = false
    }
  }

  void refresh()
  const timer = setInterval(() => void refresh(), intervalMs)

  return () => {
    stopped = true
    clearInterval(timer)
  }
}

export const proteinDesignTaskList = (): ProteinDesignTask[] =>
  Object.values($proteinDesignTasks.get().tasks).sort((a, b) => b.updatedAt - a.updatedAt)

export const resolveProteinDesignTask = (query: string): ProteinDesignTask | null => {
  const needle = query.trim()

  if (!needle) {
    return null
  }

  const exact = $proteinDesignTasks.get().tasks[needle]

  if (exact) {
    return exact
  }

  const matches = proteinDesignTaskList().filter(task => task.taskId.startsWith(needle))
  return matches.length === 1 ? matches[0] : null
}

export const focusProteinDesignTask = (taskId: string | null) =>
  $proteinDesignTasks.set({ ...$proteinDesignTasks.get(), focusedTaskId: taskId })

export const readProteinDesignTaskLog = async (taskId: string, root = taskRoot()): Promise<string> => {
  if (!safeTaskId(taskId)) {
    throw new Error(`invalid protein-design task ID: ${taskId}`)
  }
  const body = await readFile(join(root, taskId, 'worker.log'), 'utf8')
  const lines = body.split('\n')
  return lines.slice(-200).join('\n').trim() || '(worker log is empty)'
}

export const formatProteinDesignTask = (task: ProteinDesignTask): string => {
  const best = task.bestObjective == null ? '—' : task.bestObjective.toFixed(4)
  const worker = task.computeWorkerId || task.computeUrl || 'local'

  return [
    `${task.target} · ${task.taskId}`,
    `status: ${task.status}`,
    `progress: ${task.cycle}/${task.totalCycles || '?'}`,
    `phase: ${task.phase || 'setup'}`,
    `skill: ${task.selectedSkill || '—'}`,
    `best objective: ${best}`,
    `compute: ${worker}`,
    ...(task.error ? [`error: ${task.error}`] : [])
  ].join('\n')
}

export const resetProteinDesignTaskStore = () => $proteinDesignTasks.set(initialState())

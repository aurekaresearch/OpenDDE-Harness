// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import type { Theme } from '../theme.js'

import { $proteinDesignTasks, startProteinDesignTaskPolling } from '../app/proteinDesignTaskStore.js'

const MAX_VISIBLE = 3

const short = (value: string, max: number) => (value.length <= max ? value : `${value.slice(0, Math.max(1, max - 1))}…`)

const phaseLabel = (phase: string) => phase.replace(/[_-]+/g, ' ').replace(/\b\w/g, char => char.toUpperCase())

export function BackgroundTaskBar({ cols, t }: { cols: number; t: Theme }) {
  const state = useStore($proteinDesignTasks)

  useEffect(() => startProteinDesignTaskPolling(), [])

  const active = useMemo(() => {
    const tasks = Object.values(state.tasks)
      .filter(task => task.status === 'queued' || task.status === 'running')
      .sort((a, b) => b.updatedAt - a.updatedAt)

    if (state.focusedTaskId) {
      tasks.sort((a, b) => Number(b.taskId === state.focusedTaskId) - Number(a.taskId === state.focusedTaskId))
    }

    return tasks
  }, [state])

  if (!active.length) {
    return null
  }

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color={t.color.label}>
        Design Tasks ({active.length})
      </Text>
      {active.slice(0, MAX_VISIBLE).map(task => {
        const best = task.bestObjective == null ? '' : ` · best ${task.bestObjective.toFixed(4)}`
        const worker = task.computeWorkerId ? ` · ${task.computeWorkerId}` : ''
        const progress = `${task.cycle}/${task.totalCycles || '?'}`
        const line = `● ${task.target} · ${progress} · ${phaseLabel(task.phase || 'setup')}${best}${worker}`

        return (
          <Text color={t.color.statusGood} key={task.taskId} wrap="truncate-end">
            {short(line, Math.max(20, cols - 2))}
          </Text>
        )
      })}
      {active.length > MAX_VISIBLE && <Text color={t.color.muted}>+ {active.length - MAX_VISIBLE} more · /tasks</Text>}
    </Box>
  )
}

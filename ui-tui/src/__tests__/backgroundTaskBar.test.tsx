import { render } from 'ink-testing-library'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { $proteinDesignTasks, resetProteinDesignTaskStore } from '../app/proteinDesignTaskStore.js'
import { BackgroundTaskBar } from '../components/backgroundTaskBar.js'
import { DEFAULT_THEME } from '../theme.js'

afterEach(() => resetProteinDesignTaskStore())

describe('BackgroundTaskBar', () => {
  it('shows compact progress for at most three active protein-design tasks', () => {
    const tasks = Object.fromEntries(
      Array.from({ length: 4 }, (_, index) => {
        const taskId = `task-${index}`
        return [
          taskId,
          {
            bestObjective: 3.2 - index / 10,
            computeWorkerId: `zkln-0${index + 1}`,
            cycle: 10 + index,
            phase: index === 0 ? 'fold' : 'design',
            status: 'running' as const,
            target: `TARGET${index}`,
            taskId,
            totalCycles: 100,
            updatedAt: index
          }
        ]
      })
    )
    $proteinDesignTasks.set({ focusedTaskId: null, tasks })

    const view = render(<BackgroundTaskBar cols={100} t={DEFAULT_THEME} />)
    const frame = view.lastFrame() || ''

    expect(frame).toContain('Design Tasks (4)')
    expect(frame).toContain('TARGET3')
    expect(frame).toContain('13/100')
    expect(frame).toContain('best 2.9000')
    expect(frame).not.toContain('TARGET0')
    expect(frame).toContain('+ 1 more · /tasks')
    view.unmount()
  })

  it('does not occupy composer rows when no task is active', () => {
    $proteinDesignTasks.set({
      focusedTaskId: null,
      tasks: {
        done: {
          cycle: 100,
          phase: 'done',
          status: 'completed',
          target: 'DONE',
          taskId: 'done',
          totalCycles: 100,
          updatedAt: 1
        }
      }
    })

    const view = render(<BackgroundTaskBar cols={80} t={DEFAULT_THEME} />)
    expect(view.lastFrame()).toBe('')
    view.unmount()
  })
})

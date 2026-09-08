import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import {
  proteinDesignTaskList,
  refreshProteinDesignTasks,
  resetProteinDesignTaskStore,
  resolveProteinDesignTask
} from '../app/proteinDesignTaskStore.js'

let root = ''

afterEach(async () => {
  resetProteinDesignTaskStore()
  if (root) {
    await rm(root, { force: true, recursive: true })
    root = ''
  }
})

describe('proteinDesignTaskStore', () => {
  it('discovers persisted detached task snapshots and resolves short IDs', async () => {
    root = await mkdtemp(join(tmpdir(), 'opendde-protein-tasks-'))
    const directory = join(root, '1234567890abcdef')
    await mkdir(directory)
    await writeFile(
      join(directory, 'snapshot.json'),
      JSON.stringify({
        best_candidate: { objective: 2.75 },
        compute_worker_id: 'zkln-01',
        cycle: 36,
        phase: 'fold',
        selected_skill: 'cdr-point-mutation',
        status: 'running',
        target: 'CRLF2',
        task_id: '1234567890abcdef',
        total_cycles: 100
      })
    )

    await refreshProteinDesignTasks(root)

    expect(proteinDesignTaskList()).toEqual([
      expect.objectContaining({
        bestObjective: 2.75,
        computeWorkerId: 'zkln-01',
        cycle: 36,
        phase: 'fold',
        status: 'running',
        target: 'CRLF2'
      })
    ])
    expect(resolveProteinDesignTask('12345678')?.taskId).toBe('1234567890abcdef')
  })

  it('ignores malformed task directories without hiding valid tasks', async () => {
    root = await mkdtemp(join(tmpdir(), 'opendde-protein-tasks-'))
    await mkdir(join(root, 'bad'))
    await writeFile(join(root, 'bad', 'snapshot.json'), '{broken')

    await expect(refreshProteinDesignTasks(root)).resolves.toEqual([])
  })
})

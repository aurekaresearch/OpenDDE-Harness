// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.
//
// The `/resume` session picker inside the full app layout: every session
// must occupy one row with its id beside its number, and while the picker is
// open the welcome panel must not show around or behind it.

import { renderSync, stringWidth } from '@hermes/ink'
import React from 'react'
import { PassThrough } from 'stream'
import { afterEach, describe, expect, it } from 'vitest'

import type {
  AppLayoutActions,
  AppLayoutComposerProps,
  AppLayoutProps,
  AppLayoutStatusProps,
  GatewayServices
} from '../app/interfaces.js'
import type { SessionListItem } from '../gatewayTypes.js'
import type { Msg, SessionInfo } from '../types.js'

import { GatewayProvider } from '../app/gatewayContext.js'
import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { AppLayout } from '../components/appLayout.js'

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))
const ESC = String.fromCharCode(27)

// Replay the renderer's output on a fixed-size cell grid so absolute cursor
// moves, partial redraws and erases land where a terminal would put them.
const playback = (raw: string, columns: number, rows: number) => {
  const grid = Array.from({ length: rows }, () => Array<string>(columns).fill(' '))
  let row = 0
  let col = 0
  const csi = new RegExp(`^${ESC}\\[([0-9;?<>=]*)([a-zA-Z@])`)
  const osc = new RegExp(`^${ESC}\\][^\\u0007${ESC}]*(?:\\u0007|${ESC}\\\\)?`)
  const chars = [...raw]
  let i = 0

  while (i < chars.length) {
    const ch = chars[i]!

    if (ch === ESC) {
      const rest = chars.slice(i, i + 64).join('')
      const m = csi.exec(rest)

      if (m) {
        const args = m[1]!.split(';').map(v => (v === '' ? NaN : Number(v)))
        const n = Number.isNaN(args[0]!) ? 1 : args[0]!

        switch (m[2]) {
          case 'A':
            row = Math.max(0, row - n)
            break
          case 'B':
            row = Math.min(rows - 1, row + n)
            break
          case 'C':
            col = Math.min(columns - 1, col + n)
            break
          case 'D':
            col = Math.max(0, col - n)
            break
          case 'G':
            col = Math.max(0, n - 1)
            break
          case 'd':
            row = Math.max(0, n - 1)
            break
          case 'H':
          case 'f':
            row = Math.max(0, (Number.isNaN(args[0]!) ? 1 : args[0]!) - 1)
            col = Math.max(0, (args.length > 1 && !Number.isNaN(args[1]!) ? args[1]! : 1) - 1)
            break
          case 'J':
            if (args[0] === 2 || args[0] === 3) {
              for (const line of grid) {
                line.fill(' ')
              }
            } else if (Number.isNaN(args[0]!) || args[0] === 0) {
              grid[row]!.fill(' ', col)
              for (let r = row + 1; r < rows; r++) {
                grid[r]!.fill(' ')
              }
            }
            break
          case 'K':
            if (Number.isNaN(args[0]!) || args[0] === 0) {
              grid[row]!.fill(' ', col)
            } else if (args[0] === 2) {
              grid[row]!.fill(' ')
            }
            break
          default:
            break
        }

        i += [...m[0]].length
        continue
      }

      const o = osc.exec(rest)

      if (o) {
        i += [...o[0]].length
        continue
      }

      i += 2
      continue
    }

    if (ch === '\n') {
      if (row === rows - 1) {
        grid.shift()
        grid.push(Array<string>(columns).fill(' '))
      } else {
        row++
      }

      i++
      continue
    }

    if (ch === '\r') {
      col = 0
      i++
      continue
    }

    if (ch < ' ') {
      i++
      continue
    }

    const width = Math.max(1, stringWidth(ch))

    if (col < columns) {
      grid[row]![col] = ch

      if (width > 1 && col + 1 < columns) {
        grid[row]![col + 1] = ''
      }
    }

    col += width
    i++
  }

  return grid.map(line => line.join('').replace(/\s+$/, ''))
}

const SESSION: SessionInfo = {
  model: 'deepseek/deepseek-v4-flash',
  release_date: '2026-09-01',
  skills: { research: ['web_search'] },
  system_prompt: 'x'.repeat(2000),
  tools: { builtin: ['read_file', 'run', 'write_file'] },
  version: '0.0.1'
}

const SESSIONS: SessionListItem[] = Array.from({ length: 5 }, (_, i) => ({
  id: `tui:20260908_16414${i}_5c200${i}`,
  message_count: 3 + i,
  preview: `preview of session ${i} that is fairly long so it has to truncate`,
  source: 'tui',
  started_at: Date.now() / 1000 - i * 90000,
  title: ''
}))

const INTRO: Msg = { info: SESSION, kind: 'intro', role: 'system', text: '' }

const actions: AppLayoutActions = {
  answerApproval: () => {},
  answerClarify: () => {},
  answerConfirm: () => {},
  clearSelection: () => {},
  deleteSessionWithFallback: async () => false,
  onModelSelect: () => {},
  resumeById: () => {},
  setStickyPrompt: () => {}
}

const status: AppLayoutStatusProps = {
  cwdLabel: '~/repo',
  goodVibesTick: 0,
  sessionStartedAt: null,
  showStickyPrompt: false,
  statusColor: 'green',
  stickyPrompt: '',
  turnStartedAt: null
}

const makeComposer = (cols: number): AppLayoutComposerProps => ({
  cols,
  compIdx: 0,
  completions: [],
  empty: true,
  handleTextPaste: async () => null,
  input: '',
  inputBuf: [],
  pagerPageSize: 10,
  queueEditIdx: null,
  queuedDisplay: [],
  submit: () => {},
  updateInput: () => {}
})

const gwServices = {
  gw: { request: async (method: string) => (method === 'session.list' ? { sessions: SESSIONS } : null) },
  rpc: async () => null
} as unknown as GatewayServices

const makeProps = (cols: number): AppLayoutProps => ({
  actions,
  composer: makeComposer(cols),
  mouseTracking: false,
  progress: { showProgressArea: false },
  status,
  transcript: {
    historyItems: [INTRO],
    scrollRef: { current: null },
    virtualHistory: { bottomSpacer: 0, end: 1, measureRef: () => () => {}, offsets: [0], start: 0, topSpacer: 0 },
    virtualRows: [{ index: 0, key: 'intro', msg: INTRO }]
  }
})

const App = ({ cols }: { cols: number }) => (
  <GatewayProvider value={gwServices}>
    <AppLayout {...makeProps(cols)} />
  </GatewayProvider>
)

const saved = { columns: process.stdout.columns, rows: process.stdout.rows }

afterEach(() => {
  Object.assign(process.stdout, saved)
  resetOverlayState()
  resetUiState()
})

// Render through the real @hermes/ink renderer at a fixed viewport. The
// renderer lays out against the stream it is given while `useStdout` and
// `useTerminalSize` read `process.stdout`, so both get the same fake size.
// `settle` replays everything written so far, so it always yields the screen
// as a terminal would show it after the latest frame.
const mount = (columns: number, rows: number, picker: boolean) => {
  resetUiState()
  resetOverlayState()
  patchUiState({ info: SESSION, sid: 'tui:current' })
  patchOverlayState({ picker })

  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns, isTTY: true, rows })
  Object.assign(process.stdout, { columns, rows })
  Object.assign(stdin, { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })
  Object.assign(stderr, { isTTY: true })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(<App cols={columns} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  const settle = async () => {
    await delay(80)

    return playback(output, columns, rows)
  }

  return {
    done: async () => {
      const lines = await settle()
      instance.unmount()
      instance.cleanup()

      return lines
    },
    settle
  }
}

const renderAt = (columns: number, rows: number, picker: boolean) => mount(columns, rows, picker).done()

const rowOf = (lines: string[], text: string) => lines.findIndex(line => line.includes(text))

describe('session picker layout', () => {
  for (const [columns, rows] of [
    [250, 50],
    [130, 40]
  ] as [number, number][]) {
    it(`keeps every session on one row at ${columns}x${rows}`, async () => {
      const lines = await renderAt(columns, rows, true)

      expect(rowOf(lines, 'Resume Session')).toBeGreaterThanOrEqual(0)

      for (const [i, s] of SESSIONS.entries()) {
        const row = rowOf(lines, `[${s.id}]`)

        expect(row, `session ${i} id on screen`).toBeGreaterThanOrEqual(0)
        expect(lines[row], `session ${i} number beside its id`).toContain(`${i + 1}. [${s.id}]`)
        expect(lines[row], `session ${i} meta on the same row`).toContain(`(${s.message_count} msgs,`)
        expect(lines[row], `session ${i} preview on the same row`).toContain('preview of session')
      }

      // Five sessions occupy five consecutive rows.
      const first = rowOf(lines, `[${SESSIONS[0]!.id}]`)
      expect(rowOf(lines, `[${SESSIONS[4]!.id}]`)).toBe(first + 4)
      expect(lines.some(line => line.includes('1-9 quick') && line.includes('d delete'))).toBe(true)
      expect(Math.max(...lines.map(line => [...line.trimEnd()].length))).toBeLessThanOrEqual(columns)
    })

    it(`hides the welcome panel while the picker is open at ${columns}x${rows}`, async () => {
      const lines = await renderAt(columns, rows, true)
      const frame = lines.join('\n')

      expect(frame).toContain('Resume Session')
      expect(frame).not.toContain('Available Tools')
      expect(frame).not.toContain('/help')
      expect(frame).not.toMatch(/[▀▄█]/)
      expect(frame).not.toContain('v0.0.1')
      expect(frame).toContain('~/repo')

      // The pane's frame spans the terminal minus a one-column margin each side.
      const top = rowOf(lines, 'Resume Session') - 1
      expect(lines[top]).toMatch(/^ ╭─+╮$/)
      expect([...lines[top]!].length).toBe(columns - 1)
    })

    it(`shows the welcome panel again once the picker closes at ${columns}x${rows}`, async () => {
      const plain = await renderAt(columns, rows, false)
      const r = mount(columns, rows, true)
      const open = await r.settle()

      expect(open.join('\n')).toContain('Resume Session')
      expect(open.join('\n')).not.toContain('Available Tools')

      patchOverlayState({ picker: false })
      const closed = await r.done()

      expect(plain.join('\n')).toContain('Available Tools')
      expect(plain.join('\n')).toContain('/help')
      expect(closed).toEqual(plain)
    })
  }
})

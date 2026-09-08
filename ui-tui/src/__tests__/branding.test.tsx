// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { Box, renderSync } from '@hermes/ink'
import { render } from 'ink-testing-library'
import React from 'react'
import { PassThrough } from 'stream'
import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '../types.js'

import {
  lockupSize,
  OPENDDE_HARNESS_HERO_WIDTH,
  OPENDDE_HARNESS_LOGO_COMPACT_WIDTH,
  OPENDDE_HARNESS_LOGO_ROWS,
  OPENDDE_HARNESS_LOGO_WIDTH,
  openddeHarnessLogo,
  openddeHarnessLogoWord,
  OPENDDE_HARNESS_WORD_WIDTH,
  openddeHarnessWordmark,
  openddeHarnessWordmarkAt,
  resolveLockupScale,
  scaleArt,
  WELCOME_CHROME_ROWS
} from '../banner.js'
import { ArtRows, Branding, formatProvider, SessionPanel } from '../components/branding.js'
import { brandRows, planSessionPanel } from '../lib/panelLayout.js'
import { DEFAULT_THEME } from '../theme.js'

// Rows the composer and status bar hold below the welcome screen; the banner
// and panel together must fit in the rest.
const COMPOSER_ROWS = 6

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))
const ESC = String.fromCharCode(27)

// Flatten renderer output into screen text: cursor-right moves become the
// blanks they skip, every other escape sequence is dropped.
const screenText = (raw: string) =>
  raw
    .replace(new RegExp(`${ESC}\\[(\\d+)C`, 'g'), (_, n: string) => ' '.repeat(Number(n)))
    .replace(new RegExp(`${ESC}\\[[0-9;?<>=]*[a-zA-Z]`, 'g'), '')
    .replace(new RegExp(`${ESC}\\][^\\u0007]*\\u0007?`, 'g'), '')
    .replace(/\r/g, '')

// Render through the real @hermes/ink renderer at a fixed viewport so the
// size-dependent layout is deterministic. The renderer lays out against the
// stream it is given, while `useTerminalSize` (like ink's own `useStdout`)
// reads `process.stdout`, so both get the same fake size. `resize` changes
// them and emits the event the hook listens to.
const renderAt = (node: React.ReactElement, columns: number, rows: number) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  const saved = { columns: process.stdout.columns, rows: process.stdout.rows }
  let output = ''

  const setSize = (nextColumns: number, nextRows: number) => {
    Object.assign(stdout, { columns: nextColumns, rows: nextRows })
    Object.assign(process.stdout, { columns: nextColumns, rows: nextRows })
  }

  setSize(columns, rows)
  Object.assign(stdout, { isTTY: true })
  Object.assign(stdin, { isTTY: true, ref: () => {}, setRawMode: () => {}, unref: () => {} })
  Object.assign(stderr, { isTTY: true })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  const settle = async () => {
    await delay(40)

    const frame = screenText(output)
    output = ''

    return frame
  }

  return {
    done: async () => {
      const frame = await settle()
      instance.unmount()
      instance.cleanup()
      Object.assign(process.stdout, saved)

      return frame
    },
    resize: async (nextColumns: number, nextRows: number) => {
      output = ''
      setSize(nextColumns, nextRows)
      stdout.emit('resize')
      process.stdout.emit('resize')
      await delay(40)
    },
    settle
  }
}

// Longest run of block cells on any row: the double tier draws a 5-cell glyph
// row as 10 columns, the single tier as 5, the text tier draws none.
const longestBlockRun = (frame: string) => Math.max(0, ...[...frame.matchAll(/█+/g)].map(m => m[0].length))

describe('Branding', () => {
  it('does not contain hermes brand', () => {
    const { lastFrame } = render(<Branding />)
    expect(lastFrame()?.toLowerCase()).not.toContain('hermes')
  })

  it('renders without throwing when invoked with no props', () => {
    expect(() => render(<Branding />)).not.toThrow()
  })

  it('falls back to the one-line product name when even the small lockup will not fit', async () => {
    const frame = await renderAt(<Branding t={DEFAULT_THEME} />, 30, 12).done()

    expect(frame).toContain(`${DEFAULT_THEME.brand.icon}  OpenDDE Harness`)
    expect(frame).not.toContain('█')
  })

  it('spells the name out under the smallest lockup at 40x15', async () => {
    const frame = await renderAt(<Branding t={DEFAULT_THEME} />, 40, 15).done()

    expect(frame).toMatch(/[▀▄█]/)
    expect(frame).toContain('OpenDDE Harness')
  })

  it('draws the full lockup when the terminal holds it', async () => {
    const frame = await renderAt(<Branding t={DEFAULT_THEME} />, 160, 40).done()
    const lines = frame.split('\n').filter(line => line.trim())

    // The hero art's last row is blank, so the frame can be one row shorter.
    expect(lines.length).toBeGreaterThanOrEqual(lockupSize(1).rows - 1)
    expect(lines.length).toBeLessThanOrEqual(lockupSize(1).rows)
    // The double-block wordmark draws each glyph cell as two columns.
    expect(longestBlockRun(frame)).toBeGreaterThanOrEqual(10)
    expect(frame).toContain('▄')
  })

  it('scales the whole lockup down instead of dropping half of it', async () => {
    const frame = await renderAt(<Branding t={DEFAULT_THEME} />, 90, 20).done()
    const lines = frame.split('\n').filter(line => line.trim())

    expect(lines.length).toBeLessThanOrEqual(lockupSize(2).rows + 1)
    expect(lines.every(line => [...line].length <= 90)).toBe(true)
    expect(frame).toMatch(/[▀▄█]/)
  })

  it('re-renders into a different tier when the terminal is resized', async () => {
    const r = renderAt(<Branding t={DEFAULT_THEME} />, 160, 40)
    const artRows = (frame: string) => frame.split('\n').filter(line => /[▀▄█]/.test(line)).length

    expect(artRows(await r.settle())).toBeGreaterThanOrEqual(lockupSize(1).rows - 1)

    await r.resize(90, 20)
    expect(artRows(await r.settle())).toBeLessThanOrEqual(lockupSize(2).rows)

    await r.resize(30, 12)
    const small = await r.done()
    expect(small).toContain('OpenDDE Harness')
    expect(small).not.toContain('█')
  })
})

describe('brand lockup geometry', () => {
  it('pairs the hero and the wordmark at the same scale', () => {
    expect(lockupSize(1)).toMatchObject({ heroRows: 24, wordmarkRows: 16 })
    expect(lockupSize(2)).toMatchObject({ heroRows: 12, wordmarkRows: 8 })
    expect(lockupSize(4)).toMatchObject({ heroRows: 6, wordmarkRows: 4 })
  })

  it('is as wide as hero + gap + wordmark and as tall as the taller half', () => {
    for (const factor of [1, 2, 4] as const) {
      const size = lockupSize(factor)

      expect(size.cols).toBe(size.heroCols + 4 + size.wordmarkCols)
      expect(size.rows).toBe(Math.max(size.heroRows, size.wordmarkRows))
    }
  })

  it('resolveLockupScale picks the largest lockup the budget holds', () => {
    expect(resolveLockupScale(24, 130)).toBe(1)
    expect(resolveLockupScale(24, 129)).toBe(2)
    expect(resolveLockupScale(23, 130)).toBe(2)
    expect(resolveLockupScale(12, 67)).toBe(2)
    expect(resolveLockupScale(11, 67)).toBe(4)
    expect(resolveLockupScale(6, 35)).toBe(0)
    expect(resolveLockupScale(5, 36)).toBe(0)
  })

  it('scales the wordmark through the same pipeline as the hero', () => {
    const ramp = DEFAULT_THEME.yellow

    expect(openddeHarnessWordmarkAt(ramp, 1)).toHaveLength(OPENDDE_HARNESS_LOGO_ROWS)
    expect(openddeHarnessWordmarkAt(ramp, 2)).toHaveLength(8)
    expect(openddeHarnessWordmarkAt(ramp, 4)).toHaveLength(4)
    expect(
      openddeHarnessWordmarkAt(ramp, 2)
        .map(([, text]) => text)
        .join('')
    ).toMatch(/[▀▄█]/)
  })
})

describe('scaleArt', () => {
  const bitmap = ['##  ', '  ##', '####', '    ']

  it('folds two source rows into one line of half blocks', () => {
    expect(scaleArt(bitmap, 2)).toEqual(['▀▄', '▀▀'])
  })

  it('halves the height and the width of the product hero', () => {
    const half = scaleArt(['x'.repeat(OPENDDE_HARNESS_HERO_WIDTH)], 2)

    expect(half).toHaveLength(1)
    expect([...half[0]!].length).toBe(Math.ceil(OPENDDE_HARNESS_HERO_WIDTH / 2))
  })

  it('is a no-op at factor 1 and keeps every filled cell', () => {
    expect(scaleArt(bitmap, 1)).toEqual(['##', '  ##', '####', ''])
    expect(scaleArt(bitmap, 4)).toEqual(['█'])
  })
})

const SESSION: SessionInfo = {
  model: 'anthropic/claude-sonnet-4-6',
  release_date: '2026-09-01',
  skills: { research: ['web_search'] },
  system_prompt: 'x'.repeat(2000),
  tools: { file_tools: ['read_file', 'write_file'], shell_tools: ['run'] },
  version: '0.3.1'
}

// The welcome screen as appLayout draws it: the session panel alone, carrying
// the brand lockup itself (the standalone banner only shows before info lands).
const Welcome = () => (
  <Box flexDirection="column" paddingTop={1}>
    <SessionPanel info={SESSION} sid="abc123" t={DEFAULT_THEME} />
  </Box>
)

const usedRows = (frame: string) => frame.replace(/\s+$/, '').split('\n').length
const firstLine = (frame: string) => frame.split('\n').find(line => line.trim()) ?? ''

describe('welcome screen fits the terminal', () => {
  const heroCell = /[▀▄█]/

  it('draws the full lockup, hero beside wordmark, at 160x45', async () => {
    const frame = await renderAt(<Welcome />, 160, 45).done()
    const artLines = frame.split('\n').filter(line => heroCell.test(line))

    // Hero and wordmark on the same rows: the widest art line spans past the
    // hero's own width, so the two halves are side by side, not stacked.
    // The hero art's last row is blank, so the frame can be one row shorter.
    expect(artLines.length).toBeGreaterThanOrEqual(lockupSize(1).heroRows - 1)
    expect(Math.max(...artLines.map(line => [...line.trimEnd()].length))).toBeGreaterThan(lockupSize(1).heroCols + 4)
    expect(longestBlockRun(frame)).toBeGreaterThanOrEqual(10)
    expect(usedRows(frame)).toBeLessThanOrEqual(45 - COMPOSER_ROWS)
    expect(frame).toContain('Available Tools')
  })

  it('halves the lockup at 110x30', async () => {
    const frame = await renderAt(<Welcome />, 110, 30).done()
    const artLines = frame.split('\n').filter(line => heroCell.test(line))

    expect(artLines.length).toBeLessThanOrEqual(lockupSize(2).rows)
    expect(artLines.length).toBeGreaterThan(lockupSize(4).rows)
    expect(usedRows(frame)).toBeLessThanOrEqual(30 - COMPOSER_ROWS)
    expect(frame).toContain('Available Tools')
  })

  it('quarters the lockup and spells the name out at 80x22', async () => {
    const frame = await renderAt(<Welcome />, 80, 22).done()
    const artLines = frame.split('\n').filter(line => heroCell.test(line))

    expect(artLines.length).toBeLessThanOrEqual(lockupSize(4).rows)
    expect(frame).toContain('OpenDDE Harness v0.3.1 (2026-09-01)')
    expect(usedRows(frame)).toBeLessThanOrEqual(22 - COMPOSER_ROWS)
  })

  it('keeps only the text brand and the help footer at 60x16', async () => {
    const frame = await renderAt(<Welcome />, 60, 16).done()

    expect(firstLine(frame)).toContain('╭')
    expect(frame).toContain(`${DEFAULT_THEME.brand.icon}  OpenDDE Harness v0.3.1 (2026-09-01)`)
    expect(frame).not.toMatch(heroCell)
    expect(frame).toContain('/help')
    expect(usedRows(frame)).toBeLessThanOrEqual(16 - COMPOSER_ROWS)
  })

  it('fits the rows and never exceeds the columns from 60x16 to 200x60', async () => {
    for (const [columns, rows] of [
      [60, 16],
      [80, 22],
      [90, 24],
      [110, 30],
      [130, 40],
      [160, 45],
      [200, 60]
    ] as [number, number][]) {
      const frame = await renderAt(<Welcome />, columns, rows).done()
      const lines = frame.split('\n')

      expect(usedRows(frame)).toBeLessThanOrEqual(rows - COMPOSER_ROWS)
      expect(Math.max(...lines.map(line => [...line.trimEnd()].length))).toBeLessThanOrEqual(columns)
      // The brand is present either as the lockup art or as the name line.
      expect(frame).toMatch(/OpenDDE Harness|[▀▄█]/)
    }
  })

  it('re-scales the lockup when the terminal is resized', async () => {
    const r = renderAt(<Welcome />, 160, 45)
    const artRows = (frame: string) => frame.split('\n').filter(line => heroCell.test(line)).length

    expect(artRows(await r.settle())).toBeGreaterThanOrEqual(lockupSize(1).heroRows - 1)

    await r.resize(110, 30)
    expect(artRows(await r.settle())).toBeLessThanOrEqual(lockupSize(2).rows)

    await r.resize(80, 22)
    expect(artRows(await r.settle())).toBeLessThanOrEqual(lockupSize(4).rows)

    await r.resize(60, 16)
    expect(artRows(await r.done())).toBe(0)
  })
})

describe('planSessionPanel', () => {
  const sections = {
    mcp: { body: 1, open: null, present: true },
    skills: { body: 2, open: null, present: true },
    system: { body: 0, open: null, present: true },
    tools: { body: 2, open: null, present: true }
  }
  const plan = (available: number, cols = 160) => planSessionPanel({ available, cols, footerMetaLength: 30, sections })

  it('never plans a panel taller than the rows it was given', () => {
    for (let available = 5; available <= 60; available++) {
      for (const cols of [60, 80, 110, 160]) {
        const p = planSessionPanel({ available, cols, footerMetaLength: 30, sections })
        const frame = 2 * p.paddingY + 2

        expect(p.infoRows + frame + brandRows(p.lockup) + p.gap).toBeLessThanOrEqual(available)
      }
    }
  })

  it('never plans a lockup wider than the panel', () => {
    for (const cols of [40, 60, 72, 80, 110, 136, 160, 200]) {
      const p = planSessionPanel({ available: 60, cols, footerMetaLength: 30, sections })

      if (p.lockup !== 0) {
        expect(lockupSize(p.lockup).cols).toBeLessThanOrEqual(cols - 6)
      }
    }
  })

  it('shrinks the lockup as the rows shrink, then falls back to the text brand', () => {
    expect(plan(40).lockup).toBe(1)
    expect(plan(24).lockup).toBe(2)
    expect(plan(15).lockup).toBe(4)
    expect(plan(9).lockup).toBe(0)
  })

  it('puts the version in the footer for the big lockups and on a line for the small ones', () => {
    expect(plan(40)).toMatchObject({ nameLine: false, versionInFooter: true })
    expect(plan(24)).toMatchObject({ nameLine: false, versionInFooter: true })
    expect(plan(15)).toMatchObject({ nameLine: true, versionInFooter: false })
    expect(plan(9)).toMatchObject({ lockup: 0, nameLine: true, versionInFooter: false })
  })

  it('sheds spacing, then system and mcp, before the tools section', () => {
    expect(plan(60).show).toEqual({ mcp: true, skills: true, system: true, tools: true })
    expect(plan(60).gap).toBe(1)

    const tight = plan(26)
    expect(tight.gap).toBe(0)
    expect(tight.show.tools).toBe(true)
    expect(tight.open.tools).toBe(true)

    const tighter = plan(10)
    expect(tighter.show.system).toBe(false)
    expect(tighter.show.mcp).toBe(false)
    expect(tighter.show.tools).toBe(true)

    // Last to go: the working-directory line and the divider, then the section
    // bodies. The brand and the help footer always survive.
    const floor = plan(5)
    expect(floor).toMatchObject({ cwd: false, divider: false, lockup: 0, nameLine: true })
    expect(floor.open.tools).toBe(false)
  })

  it('never trades the tools or skills sections for a bigger lockup', () => {
    const p = plan(15)

    expect(p.lockup).toBe(4)
    expect(p.show.tools).toBe(true)
    expect(p.open.tools).toBe(true)
    expect(p.show.skills).toBe(true)
  })

  it('reserves at least one section line and tracks show independently of presence', () => {
    const absent = { ...sections, mcp: { body: 0, open: null, present: false } }
    const roomy = planSessionPanel({ available: 60, cols: 160, footerMetaLength: 30, sections: absent })
    const floor = planSessionPanel({ available: 5, cols: 160, footerMetaLength: 30, sections: absent })

    // An absent section costs no rows but is still reported as kept.
    expect(roomy.show.mcp).toBe(true)
    expect(roomy.sectionRows).toBe(plan(60).sectionRows - 2)
    expect(floor.sectionRows).toBe(floor.gap + 1)
    expect(roomy.infoRows).toBe(roomy.sectionRows + 2 + 1 + 2)
  })

  it('keeps a section the user opened by hand even when rows are scarce', () => {
    const p = planSessionPanel({
      available: 12,
      cols: 160,
      footerMetaLength: 30,
      sections: { ...sections, system: { body: 4, open: true, present: true } }
    })

    expect(p.show.system).toBe(true)
    expect(p.open.system).toBe(true)
  })
})

describe('SessionPanel', () => {
  it('does not render the update nudge in the panel (it lives in the status bar)', () => {
    const info: SessionInfo = {
      model: 'anthropic/claude-sonnet-4-6',
      skills: {},
      tools: {},
      update_available: true,
      update_command: 'ddeharness upgrade'
    }
    const { lastFrame } = render(<SessionPanel info={info} maxCols={80} sid="test" t={DEFAULT_THEME} />)
    expect(lastFrame()).not.toContain('Update available')
    expect(lastFrame()).not.toContain('upgrade')
  })

  it('puts the sections below the lockup, never beside it', async () => {
    const frame = await renderAt(<SessionPanel info={SESSION} sid="s" t={DEFAULT_THEME} />, 160, 45).done()
    const lines = frame.split('\n')
    const lastArt = lines.map(line => /[▀▄█]/.test(line)).lastIndexOf(true)
    const firstSection = lines.findIndex(line => line.includes('Available Tools'))

    expect(lastArt).toBeGreaterThan(0)
    expect(firstSection).toBeGreaterThan(lastArt)
    expect(lines.filter(line => /[▀▄█]/.test(line) && line.includes('Available'))).toHaveLength(0)
  })

  it('collapses the tools section by default when rows are scarce', async () => {
    const short = await renderAt(<SessionPanel info={SESSION} sid="s" t={DEFAULT_THEME} />, 80, 15).done()
    expect(short).not.toContain('read_file')

    const tall = await renderAt(<SessionPanel info={SESSION} sid="s" t={DEFAULT_THEME} />, 80, 30).done()
    expect(tall).toContain('read_file')
  })

  it('never wraps into a heap at a tiny size', async () => {
    const frame = await renderAt(<SessionPanel info={SESSION} sid="s" t={DEFAULT_THEME} />, 40, 15).done()
    const lines = frame.split('\n').filter(l => l.trim())

    expect(frame).toContain('OpenDDE Harness')
    expect(frame).toContain('/help')
    expect(lines.every(l => [...l].length <= 40)).toBe(true)
  })
})

describe('ArtRows', () => {
  it('truncates rows wider than the terminal instead of wrapping', async () => {
    const wideRow: [string, string][] = [
      ['#fff', 'A'.repeat(60)],
      ['#000', 'B'.repeat(60)]
    ]
    const frame = await renderAt(<ArtRows rows={[wideRow, wideRow]} />, 40, 10).done()
    const lines = frame.split('\n').filter(l => l.trim())

    expect(lines).toHaveLength(2)
    expect(lines.every(l => l.length <= 40)).toBe(true)
    expect(frame).not.toContain('B')
  })
})

// The welcome screen before and after session.info: the loading frame must
// already be the final frame minus the sections, so nothing above or below the
// section area moves when the handshake lands.
describe('welcome screen stability', () => {
  const Loading = () => (
    <Box flexDirection="column" paddingTop={1}>
      <SessionPanel info={null} sid={null} t={DEFAULT_THEME} />
    </Box>
  )
  // Shaped like the backend's session.info: one tools bucket, no MCP servers.
  const BACKEND_SESSION: SessionInfo = { ...SESSION, tools: { builtin: ['read_file', 'run', 'write_file'] } }
  const Loaded = () => (
    <Box flexDirection="column" paddingTop={1}>
      <SessionPanel info={BACKEND_SESSION} sid="tui:20260908_121700_abcdef" t={DEFAULT_THEME} />
    </Box>
  )
  const artCell = /[▀▄█]/
  const panelTop = (frame: string) => frame.split('\n').findIndex(line => line.includes('╭'))
  const panelBottom = (frame: string) => frame.split('\n').findIndex(line => line.includes('╰'))

  for (const [columns, rows] of [
    [130, 40],
    [80, 22]
  ] as [number, number][]) {
    it(`keeps the lockup and the panel position byte-identical at ${columns}x${rows}`, async () => {
      const loading = await renderAt(<Loading />, columns, rows).done()
      const loaded = await renderAt(<Loaded />, columns, rows).done()
      const plan = planSessionPanel({
        available: rows - WELCOME_CHROME_ROWS,
        cols: columns,
        footerMetaLength: 80,
        sections: {
          mcp: { body: 0, open: null, present: false },
          skills: { body: 0, open: null, present: true },
          system: { body: 0, open: null, present: true },
          tools: { body: 1, open: null, present: true }
        }
      })
      const top = panelTop(loading)
      const loadingLines = loading.split('\n')
      const loadedLines = loaded.split('\n')

      expect(top).toBeGreaterThanOrEqual(0)
      expect(panelTop(loaded)).toBe(top)
      expect(plan.lockupRows).toBeGreaterThan(0)

      // The border, the padding row and every lockup row, verbatim.
      const head = 1 + plan.paddingY + plan.lockupRows
      expect(loadedLines.slice(top, top + head)).toEqual(loadingLines.slice(top, top + head))
      expect(loadingLines.slice(top, top + head).join('\n')).toMatch(artCell)

      // The section area is reserved, so the frame closes on the same row.
      expect(panelBottom(loaded)).toBe(panelBottom(loading))
      expect(loaded).toContain('Available Tools')

      // More content than the reservation grows the area downward only.
      const oversized = (await renderAt(<Welcome />, columns, rows).done()).split('\n')
      expect(oversized.slice(top, top + head)).toEqual(loadingLines.slice(top, top + head))
    })
  }

  it('shows the startup message and the help footer while loading', async () => {
    const frame = await renderAt(<Loading />, 130, 40).done()

    expect(frame).toContain('loading antibody design tools')
    expect(frame).toMatch(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/)
    expect(frame).toContain('/help')
    expect(frame).not.toContain('Available Tools')
    expect(usedRows(frame)).toBeLessThanOrEqual(40 - COMPOSER_ROWS)
  })

  it('keeps the name line on its row when the version arrives at 80x22', async () => {
    const loading = await renderAt(<Loading />, 80, 22).done()
    const loaded = await renderAt(<Welcome />, 80, 22).done()
    const nameRow = (frame: string) => frame.split('\n').findIndex(line => line.includes('OpenDDE Harness'))

    expect(loading).toContain('OpenDDE Harness')
    expect(loaded).toContain('OpenDDE Harness v0.3.1 (2026-09-01)')
    expect(nameRow(loaded)).toBe(nameRow(loading))
  })

  it('fits the terminal while loading from 60x16 to 200x60', async () => {
    for (const [columns, rows] of [
      [60, 16],
      [80, 22],
      [110, 30],
      [160, 45],
      [200, 60]
    ] as [number, number][]) {
      const frame = await renderAt(<Loading />, columns, rows).done()

      expect(usedRows(frame)).toBeLessThanOrEqual(rows - COMPOSER_ROWS)
      expect(frame).toContain('/help')
    }
  })
})

const TEST_GLYPHS = new Map([
  ['01110\n10001\n10001\n11111\n10001\n10001\n10001', 'A'],
  ['11110\n10001\n10001\n10001\n10001\n10001\n11110', 'D'],
  ['11111\n10000\n10000\n11110\n10000\n10000\n11111', 'E'],
  ['01110\n10001\n10000\n10111\n10001\n10001\n01110', 'G'],
  ['10001\n10001\n10001\n11111\n10001\n10001\n10001', 'H'],
  ['10001\n11011\n10101\n10101\n10001\n10001\n10001', 'M'],
  ['10001\n11001\n10101\n10011\n10001\n10001\n10001', 'N'],
  ['01110\n10001\n10001\n10001\n10001\n10001\n01110', 'O'],
  ['11110\n10001\n10001\n11110\n10000\n10000\n10000', 'P'],
  ['11110\n10001\n10001\n11110\n10100\n10010\n10001', 'R'],
  ['01111\n10000\n10000\n01110\n00001\n00001\n11110', 'S'],
  ['11111\n00100\n00100\n00100\n00100\n00100\n00100', 'T']
])

// Decode one 7-row band of block glyphs back into letters. `cell` is the
// column width of one glyph cell (2 for the double style, 1 for single).
const decodeBlockWordmark = (lines: [string, string][], cell = 2) => {
  const block = '█'.repeat(cell)
  const rows = lines
    .map(([, text]) => {
      const cells = []

      for (let i = 0; i < text.length; i += cell) {
        cells.push(text.slice(i, i + cell) === block ? '1' : '0')
      }

      return cells.join('')
    })
    .filter(row => row.includes('1'))
  const width = Math.max(...rows.map(row => row.length))
  const blank = Array.from({ length: width }, (_, col) => rows.every(row => row[col] !== '1'))
  const glyphs: { end: number; start: number }[] = []
  let start: number | null = null

  for (let col = 0; col <= width; col++) {
    if (col < width && !blank[col] && start === null) {
      start = col
    }
    if ((col === width || blank[col]) && start !== null) {
      glyphs.push({ end: col - 1, start })
      start = null
    }
  }

  return glyphs
    .map((glyph, index) => {
      const signature = rows.map(row => row.slice(glyph.start, glyph.end + 1)).join('\n')
      const letter = TEST_GLYPHS.get(signature) ?? '?'
      const previous = glyphs[index - 1]
      const gap = previous ? glyph.start - previous.end - 1 : 0

      return `${gap >= 3 ? ' ' : ''}${letter}`
    })
    .join('')
}

// The stacked mark is two 8-row bands (7 glyph rows + 1 blank), one per word.
const decodeStacked = (lines: [string, string][], cell = 2) =>
  [lines.slice(0, 8), lines.slice(8, 16)].map(band => decodeBlockWordmark(band, cell))

// The wordmark itself is width-independent; exercise the pure builders so these
// don't depend on ink-testing's terminal columns.
describe('banner wordmark', () => {
  const ramp = DEFAULT_THEME.yellow

  it('stacks OPENDDE over HARNESS in the double-width style', () => {
    expect(decodeStacked(openddeHarnessWordmark(ramp, 'double'))).toEqual(['OPENDDE', 'HARNESS'])
  })

  it('stacks OPENDDE over HARNESS in the single-width style', () => {
    expect(decodeStacked(openddeHarnessWordmark(ramp, 'single'), 1)).toEqual(['OPENDDE', 'HARNESS'])
  })

  it('openddeHarnessLogo is the double-width stacked wordmark', () => {
    const lines = openddeHarnessLogo(ramp)
    expect(lines.length).toBe(OPENDDE_HARNESS_LOGO_ROWS)
    expect(lines.length).toBe(16)
    expect(decodeStacked(lines)).toEqual(['OPENDDE', 'HARNESS'])
    expect(lines.map(([, text]) => text).join('')).not.toContain('AGENT')
  })

  it('spreads the ramp gradient top to bottom across the stacked mark', () => {
    const colors = openddeHarnessWordmark([ramp[1]!, ramp[2]!, ramp[3]!, ramp[4]!]).map(([c]) => c)
    expect(colors.slice(0, 4).every(c => c === ramp[1])).toBe(true)
    expect(colors.slice(12, 16).every(c => c === ramp[4])).toBe(true)
  })

  it('openddeHarnessLogoWord renders just OPENDDE within one-word width', () => {
    const lines = openddeHarnessLogoWord(ramp)
    expect(lines.length).toBe(8)
    expect(decodeBlockWordmark(lines)).toBe('OPENDDE')
    const maxWidth = Math.max(...lines.map(([, text]) => [...text].length))
    expect(maxWidth).toBeLessThanOrEqual(OPENDDE_HARNESS_WORD_WIDTH)
  })

  it('the stacked wordmark is 82 columns double-width and 41 single-width', () => {
    expect(OPENDDE_HARNESS_LOGO_WIDTH).toBe(82)
    expect(OPENDDE_HARNESS_LOGO_COMPACT_WIDTH).toBe(41)
    expect(OPENDDE_HARNESS_WORD_WIDTH).toBe(82)
  })
})

describe('formatProvider', () => {
  it('returns LUT value for known anthropic slug', () => {
    expect(formatProvider('anthropic', 'claude-sonnet-4-6')).toBe('Anthropic')
  })

  it('parses model_id prefix when slug is "auto" (LiteLLM dispatch)', () => {
    expect(formatProvider('auto', 'openrouter/qwen/qwen3.6-plus')).toBe('OpenRouter')
  })

  it('returns LUT value for qwen slug', () => {
    expect(formatProvider('qwen', 'qwen-max')).toBe('Qwen')
  })

  it('returns em-dash fallback when slug empty and model_id has no slash prefix', () => {
    expect(formatProvider('', 'sonnet')).toBe('—')
  })

  it('returns em-dash fallback when slug is "auto" and model_id is empty', () => {
    expect(formatProvider('auto', '')).toBe('—')
  })

  it('returns canonical OpenAI (not Openai) for openai slug', () => {
    expect(formatProvider('openai', 'gpt-4')).toBe('OpenAI')
  })

  it('falls back to capitalize for unknown providers', () => {
    expect(formatProvider('xyz', 'xyz-foo')).toBe('Xyz')
  })
})

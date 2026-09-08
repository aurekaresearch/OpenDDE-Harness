// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { describe, expect, it } from 'vitest'

import { LONG_RUN_CHARMS } from '../content/charms.js'
import { FACES } from '../content/faces.js'
import { randomFortune } from '../content/fortunes.js'
import { HOTKEYS } from '../content/hotkeys.js'
import { PLACEHOLDERS } from '../content/placeholders.js'
import { TOOL_VERBS, VERBS } from '../content/verbs.js'
import { ROLE } from '../domain/roles.js'
import { ZERO } from '../domain/usage.js'
import { INTERPOLATION_RE } from '../protocol/interpolation.js'
import { DEFAULT_THEME } from '../theme.js'

describe('constants', () => {
  it('ZERO', () => expect(ZERO).toEqual({ calls: 0, input: 0, output: 0, total: 0 }))

  it('string arrays are populated', () => {
    for (const arr of [FACES, PLACEHOLDERS, VERBS]) {
      expect(arr.length).toBeGreaterThan(0)
      arr.forEach(s => expect(typeof s).toBe('string'))
    }
  })

  // The TUI ships an antibody-design product, so none of its stock copy may
  // read as a generic coding agent.
  it('stock copy carries no generic-agent vocabulary', () => {
    const banned = /refactor|codebase|auth module|lint|commit message|unit test|write a test/i
    const fortunes = Array.from({ length: 200 }, () => randomFortune())

    for (const line of [
      ...PLACEHOLDERS,
      ...LONG_RUN_CHARMS,
      ...fortunes,
      DEFAULT_THEME.brand.welcome,
      DEFAULT_THEME.brand.goodbye,
      DEFAULT_THEME.brand.helpHeader,
      ...HOTKEYS.map(([, desc]) => desc)
    ]) {
      expect(line).not.toMatch(banned)
    }
  })

  it('placeholders are a wide set of things a scientist would actually ask', () => {
    expect(PLACEHOLDERS.length).toBeGreaterThanOrEqual(25)
    expect(new Set(PLACEHOLDERS).size).toBe(PLACEHOLDERS.length)
    expect(PLACEHOLDERS.some(p => /VHH|CRLF2/.test(p))).toBe(true)
    expect(PLACEHOLDERS.some(p => /design task/i.test(p))).toBe(true)
    expect(PLACEHOLDERS.some(p => /epitope/i.test(p))).toBe(true)
    expect(PLACEHOLDERS.some(p => /developability/i.test(p))).toBe(true)
    expect(PLACEHOLDERS.some(p => /GPU/.test(p))).toBe(true)

    for (const placeholder of PLACEHOLDERS) {
      // Sentence case, and never a slash command or a file path.
      expect(placeholder[0]).toBe(placeholder[0]?.toUpperCase())
      expect(placeholder).not.toMatch(/^\//)
      expect(placeholder).not.toMatch(/\.(ya?ml|json|py|ts|md)\b/)
      expect(placeholder).not.toMatch(/\p{Extended_Pictographic}/u)
    }
  })

  it('busy verbs are short lowercase gerunds from the lab', () => {
    expect(VERBS.length).toBeGreaterThanOrEqual(25)
    expect(new Set(VERBS).size).toBe(VERBS.length)
    expect(VERBS.some(v => /fold/.test(v))).toBe(true)
    expect(VERBS.some(v => /CDR|paratope|epitope/.test(v))).toBe(true)

    for (const verb of VERBS) {
      expect(verb.split(' ')[0]).toMatch(/ing$/)
      expect(verb[0]).toBe(verb[0]?.toLowerCase())
      // The status bar reserves the longest verb plus an ellipsis.
      expect(verb.length).toBeLessThanOrEqual(20)
      expect(verb).not.toContain('…')
    }
  })

  it('HOTKEYS are [key, desc] pairs', () => {
    HOTKEYS.forEach(([k, d]) => {
      expect(typeof k).toBe('string')
      expect(typeof d).toBe('string')
    })
  })

  it('documents Ctrl/Cmd+L as non-destructive redraw', () => {
    const hotkey = HOTKEYS.find(([k]) => k.endsWith('+L'))
    expect(hotkey).toBeDefined()
    expect(hotkey?.[1]).toBe('redraw / repaint')
  })

  it('TOOL_VERBS maps known tools (verb-only, no emoji)', () => {
    expect(TOOL_VERBS.terminal).toBe('terminal')
    expect(TOOL_VERBS.read_file).toBe('reading')
  })

  it('INTERPOLATION_RE matches {!cmd}', () => {
    INTERPOLATION_RE.lastIndex = 0
    expect(INTERPOLATION_RE.test('{!date}')).toBe(true)

    INTERPOLATION_RE.lastIndex = 0
    expect(INTERPOLATION_RE.test('plain')).toBe(false)
  })

  it('ROLE produces glyph/body/prefix per role', () => {
    for (const role of ['assistant', 'system', 'tool', 'user'] as const) {
      expect(ROLE[role](DEFAULT_THEME)).toHaveProperty('glyph')
    }
  })
})

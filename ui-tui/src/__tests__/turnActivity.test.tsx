// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { stringWidth } from '@hermes/ink'
import { render } from 'ink-testing-library'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { indicatorWidth } from '../components/appChrome.js'
import { activityDetails, breathAt, retryVerb, TurnActivity } from '../components/turnActivity.js'
import { FACES } from '../content/faces.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

afterEach(() => {
  resetUiState()
  resetTurnState()
})

describe('activityDetails', () => {
  it('shows elapsed time, tokens and the interrupt hint, like Claude Code', () => {
    expect(activityDetails({ elapsedMs: 130_000, tokens: 1234 })).toBe('(2m 10s · ↓ 1.2k tokens · Ctrl+C to interrupt)')
  })

  it('leaves out what it does not have yet', () => {
    expect(activityDetails({ elapsedMs: null, tokens: 0 })).toBe('(Ctrl+C to interrupt)')
  })
})

describe('indicatorWidth', () => {
  it('reserves the widest frame of the style so the line never shifts', () => {
    const width = indicatorWidth('kaomoji', '*')

    expect(width).toBe(Math.max(...FACES.map(face => stringWidth(face))))
    expect(indicatorWidth('ascii', '*')).toBe(1)
  })

  it('keeps every face within two columns of the others', () => {
    const widths = FACES.map(face => stringWidth(face))

    expect(Math.max(...widths) - Math.min(...widths)).toBeLessThanOrEqual(2)
    expect(new Set(FACES).size).toBe(FACES.length)
  })
})

describe('breathAt', () => {
  it('dims, settles, glows and settles again on every four ticks', () => {
    expect([0, 1, 2, 3, 4].map(breathAt)).toEqual(['dim', 'normal', 'bright', 'normal', 'dim'])
  })
})

describe('TurnActivity', () => {
  it('renders nothing when idle and the activity line when busy', () => {
    const idle = stripAnsi(render(<TurnActivity t={DEFAULT_THEME} turnStartedAt={null} />).lastFrame() ?? '')
    expect(idle.trim()).toBe('')

    patchUiState({ busy: true })
    patchTurnState({ outputTokens: 1500 })
    const frame = stripAnsi(
      render(<TurnActivity t={DEFAULT_THEME} turnStartedAt={Date.now() - 3_000} />).lastFrame() ?? ''
    )

    expect(frame).toContain('↓ 1.5k tokens')
    expect(frame).toContain('Ctrl+C to interrupt')
    expect(frame).toMatch(/…/)
  })

  it('reads the retry in place of the verb while a call is re-run, like Codex', () => {
    patchUiState({ busy: true })
    patchTurnState({ outputTokens: 90, retry: { attempt: 2, reason: 'network', total: 4 } })
    const frame = stripAnsi(
      render(<TurnActivity t={DEFAULT_THEME} turnStartedAt={Date.now() - 24_000} />).lastFrame() ?? ''
    )

    expect(frame).toContain('retrying 2/4 (network)… (24s · ↓ 90 tokens · Ctrl+C to interrupt)')
    expect(retryVerb({ attempt: 1, reason: '', total: 4 })).toBe('retrying 1/4')
  })
})

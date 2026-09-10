// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo, useEffect, useState } from 'react'

import type { TurnRetry } from '../app/turnStore.js'
import type { Theme } from '../theme.js'

import { useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { VERBS } from '../content/verbs.js'
import { fmtDuration } from '../domain/messages.js'
import { fmtTokens } from '../lib/subagentTree.js'
import { indicatorWidth, renderIndicator } from './appChrome.js'

// Claude Code's parenthesis, nothing more: how long this turn has run, how
// many tokens it has produced (the vendor's count for finished calls plus an
// estimate of the one streaming, thinking included), how to stop it.
export const activityDetails = ({ elapsedMs, tokens }: { elapsedMs: null | number; tokens: number }): string => {
  const parts = [
    elapsedMs === null ? '' : fmtDuration(elapsedMs),
    tokens > 0 ? `↓ ${fmtTokens(tokens)} tokens` : '',
    'Ctrl+C to interrupt'
  ]

  return `(${parts.filter(Boolean).join(' · ')})`
}

export const retryVerb = ({ attempt, reason, total }: TurnRetry): string =>
  `retrying ${attempt}/${total}${reason ? ` (${reason})` : ''}`

// One breath: the indicator dims, brightens, glows, and settles, as Claude
// Code's spinner does. Faces change on their own slower cadence.
const BREATH_MS = 400
const BREATH = ['dim', 'normal', 'bright', 'normal'] as const

export const breathAt = (tick: number): (typeof BREATH)[number] => BREATH[tick % BREATH.length] ?? 'normal'

// The line above the composer while a turn runs, shaped like Claude Code's:
// the indicator breathes in a fixed-width column so its frames cannot shift
// the rest, one themed verb is drawn for the turn and stays, then elapsed
// time and token count. Tools and commands are shown in the transcript, not
// here.
export const TurnActivity = memo(function TurnActivity({
  t,
  turnStartedAt
}: {
  t: Theme
  turnStartedAt: null | number
}) {
  const ui = useStore($uiState)
  const outputTokens = useTurnSelector(state => state.outputTokens)
  const retry = useTurnSelector(state => state.retry)
  const [tick, setTick] = useState(0)
  const [verb, setVerb] = useState(() => VERBS[Math.floor(Math.random() * VERBS.length)] ?? 'working')
  const [now, setNow] = useState(() => Date.now())
  const brandMark = ui.theme.brand.icon
  const { intervalMs, showVerb } = renderIndicator(ui.indicatorStyle, 0, brandMark)
  const busy = ui.busy

  useEffect(() => {
    if (!busy) {
      return
    }

    // One verb per turn, drawn when the turn starts.
    setVerb(VERBS[Math.floor(Math.random() * VERBS.length)] ?? 'working')
    setNow(Date.now())
    const glyph = setInterval(() => setTick(n => n + 1), Math.min(BREATH_MS, intervalMs))
    const clock = setInterval(() => setNow(Date.now()), 1000)

    return () => {
      clearInterval(glyph)
      clearInterval(clock)
    }
  }, [busy, intervalMs])

  if (!busy) {
    return null
  }

  // A slow style (faces every 2.5 s) still breathes every 400 ms: the frame
  // advances once per interval while the breath runs on every tick.
  const step = Math.min(BREATH_MS, intervalMs)
  const { frame } = renderIndicator(ui.indicatorStyle, Math.floor((tick * step) / intervalMs), brandMark)
  const breath = breathAt(tick)
  const headline = showVerb ? `${retry ? retryVerb(retry) : verb}… ` : ''
  const details = activityDetails({
    elapsedMs: turnStartedAt ? now - turnStartedAt : null,
    tokens: outputTokens
  })

  return (
    <Box height={1}>
      <Box flexShrink={0} width={indicatorWidth(ui.indicatorStyle, brandMark)}>
        <Text bold={breath === 'bright'} color={t.color.accent} dimColor={breath === 'dim'}>
          {frame}
        </Text>
      </Box>
      <Text wrap="truncate-end">
        <Text bold={breath === 'bright'} color={retry ? t.color.warn : t.color.accent} dimColor={breath === 'dim'}>
          {headline ? ` ${headline}` : ' '}
        </Text>
        <Text color={t.color.muted}>{details}</Text>
      </Text>
    </Box>
  )
})

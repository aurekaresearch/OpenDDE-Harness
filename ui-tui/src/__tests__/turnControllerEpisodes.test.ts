// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { afterEach, describe, expect, it } from 'vitest'

import type { Msg } from '../types.js'

import { turnController } from '../app/turnController.js'
import { getTurnState } from '../app/turnStore.js'
import { patchUiState } from '../app/uiStore.js'

afterEach(() => {
  patchUiState({ transcript: 'legacy' })
  turnController.reset()
})

describe('turnController episodes commit', () => {
  it('buckets a turn into episodes and commits one episodes message', () => {
    patchUiState({ transcript: 'episodes' })
    turnController.reset()

    // Call 0: reason, then read two files.
    turnController.recordEpisodeStart(0)
    turnController.recordReasoningDelta('planning the reads')
    turnController.recordToolStart('a', 'read_file', 'src/approve.go')
    turnController.recordToolComplete('a', 'read_file', undefined, 'package device', 0.1)
    turnController.recordToolStart('b', 'read_file', 'src/token.go')
    turnController.recordToolComplete('b', 'read_file', undefined, 'package device 2', 0.1)
    // Call 1: the stop call — no tools, just the final answer.
    turnController.recordEpisodeStart(1)

    const { finalMessages } = turnController.recordMessageComplete({ text: 'FINAL ANSWER' })

    const epMsg = finalMessages.find(m => m.kind === 'episodes')
    expect(epMsg).toBeTruthy()
    expect(epMsg!.text).toBe('FINAL ANSWER')
    // The stop call (no tools, no reasoning) is dropped; only the work step remains.
    expect(epMsg!.episodes).toHaveLength(1)
    const [work] = epMsg!.episodes!
    expect(work!.tools.map(t => t.name)).toEqual(['read_file', 'read_file'])
    expect(work!.tools.every(t => t.ok)).toBe(true)
    expect(work!.reasoning).toContain('planning the reads')
  })

  it('records +/- line counts from an inline-diff edit', () => {
    patchUiState({ transcript: 'episodes' })
    turnController.reset()

    turnController.recordEpisodeStart(0)
    turnController.recordToolStart('e', 'edit_file', 'notes.md')
    turnController.recordInlineDiffToolComplete('--- a\n+++ b\n-old\n+new1\n+new2', 'e', 'edit_file', undefined, 0.1)
    turnController.recordEpisodeStart(1)

    const { finalMessages } = turnController.recordMessageComplete({ text: 'done' })
    const editTool = finalMessages.find(m => m.kind === 'episodes')!.episodes![0]!.tools[0]!

    expect(editTool.added).toBe(2)
    expect(editTool.removed).toBe(1)
  })

  it('interrupt in episodes mode commits one collapsed episodes message, not raw segments', () => {
    patchUiState({ transcript: 'episodes' })
    turnController.reset()

    turnController.recordEpisodeStart(0)
    turnController.recordReasoningDelta('thinking about it')
    turnController.recordToolStart('a', 'web_search', 'gtm agent')
    turnController.recordToolComplete('a', 'web_search', 'boom', undefined, 0.1)

    const appended: Msg[] = []
    turnController.finalizeInterruptedTurn({ appendMessage: m => appended.push(m) })

    // Exactly one collapsed episodes message — no raw per-segment dump (the bug:
    // a Ctrl+C used to append every expanded thinking/tool segment).
    expect(appended.every(m => m.kind === 'episodes')).toBe(true)
    const epMsgs = appended.filter(m => m.kind === 'episodes')
    expect(epMsgs).toHaveLength(1)
    expect(epMsgs[0]!.episodes!).toHaveLength(1)
    expect(epMsgs[0]!.episodes![0]!.tools[0]!.name).toBe('web_search')
    expect(epMsgs[0]!.text).toContain('[interrupted]')
  })

  it('interrupt with no episode.start still keeps the legacy trail', () => {
    // An older gateway never emits episode.start. The completion path already
    // falls back to the segment trail in that case; the interrupt path must
    // too, or a Ctrl+C silently drops everything the turn had accrued.
    patchUiState({ transcript: 'episodes' })
    turnController.reset()

    turnController.recordReasoningDelta('reasoned without any episode boundary')
    turnController.recordToolStart('a', 'web_search', 'gtm agent')
    turnController.recordToolComplete('a', 'web_search', undefined, '12 results', 0.2)

    expect(turnController.episodes).toHaveLength(0)

    const appended: Msg[] = []
    turnController.finalizeInterruptedTurn({ appendMessage: m => appended.push(m) })

    expect(appended.some(m => m.kind === 'episodes')).toBe(false)
    // The legacy trail survives with the reasoning and the tool line on it...
    const trail = appended.find(m => m.kind === 'trail')
    expect(trail).toBeTruthy()
    expect(trail!.thinking).toContain('without any episode boundary')
    expect(trail!.tools!.join(' ')).toContain('Web Search')
    expect(trail!.tools!.join(' ')).toContain('12 results')
    // ...and the interruption is still recorded.
    expect(appended.some(m => (m.text ?? '').includes('[interrupted]'))).toBe(true)
  })

  it('legacy mode is fully inert: no episodes accrue and no episodes message', () => {
    patchUiState({ transcript: 'legacy' })
    turnController.reset()

    turnController.recordEpisodeStart(0)
    turnController.recordToolStart('a', 'exec', 'ls')
    turnController.recordToolComplete('a', 'exec', undefined, 'out', 0.1)

    // recordEpisodeStart is gated on episodes mode, so nothing accrued.
    expect(turnController.episodes).toHaveLength(0)

    const { finalMessages } = turnController.recordMessageComplete({ text: 'hi' })

    expect(finalMessages.some(m => m.kind === 'episodes')).toBe(false)
  })
})

describe('turnController retry', () => {
  it("discards the current call's live text and accrual, then re-fills from the re-run", () => {
    patchUiState({ transcript: 'episodes' })
    turnController.reset()

    turnController.recordEpisodeStart(0)
    turnController.recordReasoningDelta('first try')
    turnController.recordToolStart('search1', 'web_search', 'searching')
    turnController.recordMessageDelta({ text: 'partial answer that was cut' })
    turnController.recordRetry({ attempt: 2, discard: true, reason: 'network', total: 4 })
    turnController.recordReasoningDelta('second try')
    turnController.recordMessageDelta({ text: 'whole answer' })

    expect(turnController.bufRef).toBe('whole answer')
    expect(getTurnState().tools).toEqual([])
    expect(turnController.episodes[0]!.reasoning).toBe('second try')
    expect(getTurnState().activity.some(a => a.text.includes('retrying 2/4: network'))).toBe(true)
    expect(getTurnState().retry).toBeNull()

    const { finalMessages } = turnController.recordMessageComplete({ text: 'whole answer' })
    const epMsg = finalMessages.find(m => m.kind === 'episodes')!

    expect(epMsg.text).toBe('whole answer')
    expect(epMsg.episodes![0]!.reasoning).toBe('second try')
  })

  it("drops the discarded call's trail segments but keeps earlier calls' work", () => {
    turnController.reset()
    turnController.startMessage()

    turnController.recordEpisodeStart(0)
    turnController.recordToolStart('a', 'read_file', 'src/a.go')
    turnController.recordToolComplete('a', 'read_file', undefined, 'package a', 0.1)
    turnController.recordEpisodeStart(1)
    turnController.recordReasoningDelta('discarded reasoning')
    turnController.recordMessageDelta({ text: 'cut' })
    turnController.recordRetry({ attempt: 2, discard: true, reason: 'network', total: 4 })
    turnController.recordReasoningDelta('kept reasoning')

    const trail = turnController.segmentMessages.map(m => m.thinking ?? m.tools?.join(',') ?? m.text)

    expect(trail.some(t => t?.includes('discarded reasoning'))).toBe(false)
    expect(trail.some(t => t?.includes('kept reasoning'))).toBe(true)
    expect(trail.some(t => t?.includes('src/a.go'))).toBe(true)
  })

  it('a reasoning-only previous call keeps its segment through a discarding retry', () => {
    turnController.reset()
    turnController.startMessage()

    turnController.recordEpisodeStart(0)
    turnController.recordReasoningDelta('earlier call reasoning')
    turnController.recordEpisodeStart(1)
    turnController.recordReasoningDelta('discarded reasoning')
    turnController.recordRetry({ attempt: 2, discard: true, reason: 'network', total: 4 })
    turnController.recordReasoningDelta('kept reasoning')

    const thinking = turnController.segmentMessages.map(m => m.thinking ?? '')

    expect(thinking.some(t => t.includes('earlier call reasoning') && !t.includes('discarded'))).toBe(true)
    expect(thinking.some(t => t.includes('discarded reasoning'))).toBe(false)
    expect(thinking.some(t => t.includes('kept reasoning'))).toBe(true)
  })

  it('keeps the live text when nothing had been shown before the retry', () => {
    turnController.reset()
    turnController.recordMessageDelta({ text: 'kept' })
    turnController.recordRetry({ attempt: 2, discard: false, reason: 'network', total: 4 })

    expect(turnController.bufRef).toBe('kept')
  })

  it('holds the retry for the activity line until the re-run delivers', () => {
    turnController.reset()
    turnController.recordRetry({ attempt: 2, discard: false, reason: 'network', total: 4 })

    expect(getTurnState().retry).toEqual({ attempt: 2, reason: 'network', total: 4 })

    turnController.recordRetry({ attempt: 3, discard: false, reason: 'timeout', total: 4 })

    expect(getTurnState().retry).toEqual({ attempt: 3, reason: 'timeout', total: 4 })

    turnController.recordToolStart('t1', 'read_file', 'reading')

    expect(getTurnState().retry).toBeNull()

    turnController.recordRetry({ attempt: 2, discard: false, reason: 'network', total: 4 })
    turnController.reset()

    expect(getTurnState().retry).toBeNull()
  })
})

describe('turnController output tokens', () => {
  it('estimates the call in flight and replaces it with the vendor count when the call ends', () => {
    turnController.reset()
    turnController.startMessage()

    turnController.recordReasoningDelta('x'.repeat(400))
    turnController.recordMessageDelta({ text: 'y'.repeat(400) })
    expect(getTurnState().outputTokens).toBe(200)

    turnController.recordUsage(950, 700)
    expect(getTurnState().outputTokens).toBe(950)

    turnController.recordMessageDelta({ text: 'z'.repeat(40) })
    expect(getTurnState().outputTokens).toBe(960)
  })
})
